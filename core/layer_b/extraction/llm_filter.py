"""Ollama-assisted signature filtering."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import List, Sequence

import numpy as np

from core.layer_b.extraction.vectorise import pattern_shape_stats

log = logging.getLogger(__name__)


def _quantile(values: Sequence[float], q: float, default: float) -> float:
    if not values:
        return default
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _suspicion_score(signature: dict, role: str) -> float:
    """Higher score = more likely to be an artifact that needs review."""
    pattern = str(signature.get("pattern", ""))
    stats = pattern_shape_stats(pattern)

    score = 0.0
    # Low alpha ratio → likely contains formatting junk
    if stats["alpha_ratio"] < 0.85:
        score += (0.85 - stats["alpha_ratio"]) * 5.0
    # High punctuation → template/formatting artifact
    if stats["punct_ratio"] > 0.05:
        score += stats["punct_ratio"] * 4.0
    # Edge punctuation → wrapping artifact
    if stats["edge_punct_count"] > 0:
        score += stats["edge_punct_count"] * 1.5
    # High capitalization → named entities or headers
    if stats["capitalized_token_ratio"] > 0.3:
        score += stats["capitalized_token_ratio"] * 3.0
    # High stopword ratio → generic filler
    if stats["stopword_ratio"] > 0.5:
        score += stats["stopword_ratio"] * 2.0
    # Low unique token ratio → repetitive pattern
    if stats["unique_token_ratio"] < 0.8:
        score += (1.0 - stats["unique_token_ratio"]) * 2.0
    # Short mean token length → likely fragments
    if stats["mean_token_len"] < 3.5:
        score += (3.5 - stats["mean_token_len"]) * 1.0
    # High digit ratio → data/benchmark artifact
    if stats["digit_ratio"] > 0.0:
        score += stats["digit_ratio"] * 3.0
    # Low support → more likely to be noise
    support = float(signature.get("support", 0))
    if support < 50:
        score += (50 - support) * 0.02
    # Role-specific: malicious sigs with high safe support are suspect
    if role == "malicious":
        safe_support = float(signature.get("safe_support", 0))
        if safe_support > 0:
            score += safe_support * 0.5
    return score


def select_signatures_for_llm_review(signatures: List[dict], *, role: str, max_review: int) -> List[dict]:
    """Select the most suspicious signatures for LLM review."""
    if not signatures or max_review <= 0:
        return []

    scored = [(_suspicion_score(sig, role), sig) for sig in signatures]
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [sig for _, sig in scored[:max_review]]

    top_score = scored[0][0] if scored else 0.0
    cutoff_score = scored[min(max_review, len(scored)) - 1][0] if scored else 0.0
    log.info(
        "Selected %d/%d %s signatures for LLM review (suspicion score range: %.2f - %.2f)",
        len(selected),
        len(signatures),
        role,
        cutoff_score,
        top_score,
    )
    return selected


def _build_prompt(role: str, batch: List[dict]) -> str:
    role_guidance = {
        "safe": (
            "These are SAFE-ALLOW patterns — they let prompts bypass deeper analysis. "
            "A bad safe pattern that matches attack text causes a false negative. "
            "DROP patterns that: are narrow topic/entity-specific (e.g. a specific product name, "
            "benchmark preamble, template header); contain formatting artifacts (brackets, pipes, "
            "special chars); are too generic/stopword-heavy to reliably indicate safety; "
            "look like dataset boilerplate rather than genuine user language."
        ),
        "malicious": (
            "These are MALICIOUS-BLOCK patterns — they flag prompts as attacks. "
            "A bad malicious pattern that matches benign text causes a false positive. "
            "DROP patterns that: are generic phrases found in normal conversation; "
            "contain benchmark/evaluation formatting artifacts; are assistant-side response "
            "wrappers or persona preambles; are narrow topical fragments unrelated to attacks; "
            "contain structural tokens (numbering, bullet markers, template variables)."
        ),
    }[role]

    lines = [
        "You are a strict quality filter for a prompt-injection detection system's signature rules.",
        f"Role: {role.upper()}",
        role_guidance,
        "",
        "Review each candidate critically. Drop any that would not generalize well to unseen data.",
        "When in doubt about a pattern's quality, DROP it — false positives/negatives are costly.",
        "",
        'Return ONLY valid JSON: {"drop_ids": [1, 2, ...]}',
        'If all candidates are good, return: {"drop_ids": []}',
        "",
        "Candidates:",
    ]

    for idx, sig in enumerate(batch, start=1):
        pattern = str(sig.get("pattern", ""))
        support = int(sig.get("support", 0))
        other_support = int(sig.get("safe_support", sig.get("malicious_support", 0)))
        precision = float(sig.get("precision", sig.get("safe_precision", 0.0)))
        stats = pattern_shape_stats(pattern)
        lines.append(
            json.dumps(
                {
                    "id": idx,
                    "pattern": pattern,
                    "support": support,
                    "cross_class_support": other_support,
                    "precision": round(precision, 4),
                    "token_count": int(stats["token_count"]),
                    "alpha_ratio": round(stats["alpha_ratio"], 3),
                    "punct_ratio": round(stats["punct_ratio"], 3),
                    "stopword_ratio": round(stats["stopword_ratio"], 3),
                    "capitalized_ratio": round(stats["capitalized_token_ratio"], 3),
                },
                ensure_ascii=True,
            )
        )

    return "\n".join(lines)


def _call_ollama(model: str, prompt: str, timeout_s: int) -> str | None:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 400,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return str(payload.get("response", "")).strip()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        log.warning("Ollama signature review failed: %s", exc)
        return None


def _parse_drop_ids(response_text: str, batch_size: int) -> List[int]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return []
        payload = json.loads(match.group(0))

    raw_ids = payload.get("drop_ids", [])
    if not isinstance(raw_ids, list):
        return []

    normalized: List[int] = []
    for item in raw_ids:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= batch_size:
            normalized.append(idx)
    return normalized


def apply_llm_signature_filter(
    signatures: List[dict],
    *,
    role: str,
    model: str,
    max_review: int,
    batch_size: int,
    timeout_s: int = 60,
) -> List[dict]:
    """Apply LLM review on a broad set of suspicious signatures."""
    review_subset = select_signatures_for_llm_review(
        signatures,
        role=role,
        max_review=max_review,
    )
    if not review_subset:
        return signatures

    dropped_patterns: set[str] = set()
    n_batches = (len(review_subset) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(review_subset), batch_size), start=1):
        batch = review_subset[start:start + batch_size]
        if batch_idx % 10 == 1 or batch_idx == n_batches:
            log.info("  LLM review %s batch %d/%d …", role, batch_idx, n_batches)
        response_text = _call_ollama(model, _build_prompt(role, batch), timeout_s)
        if not response_text:
            continue
        for idx in _parse_drop_ids(response_text, len(batch)):
            pattern = str(batch[idx - 1].get("pattern", ""))
            if pattern:
                dropped_patterns.add(pattern)

    if not dropped_patterns:
        log.info("LLM filter kept all reviewed %s signatures", role)
        return signatures

    kept = [sig for sig in signatures if str(sig.get("pattern", "")) not in dropped_patterns]
    log.info(
        "LLM filter dropped %d %s signatures; %d → %d remain",
        len(dropped_patterns),
        role,
        len(signatures),
        len(kept),
    )
    return kept