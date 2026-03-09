"""Bounded Ollama-assisted signature filtering.

The statistical pipeline remains primary. This module only reviews a small,
high-impact subset of signatures after extraction and can drop obvious
artifacts that survive numeric filters.
"""

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


def _review_score(signature: dict, role: str, support_floor: float) -> float:
    pattern = str(signature.get("pattern", ""))
    support = float(signature.get("support", 0))
    stats = pattern_shape_stats(pattern)

    score = support
    if support >= support_floor:
        score += support * 0.30
    if stats["token_count"] >= 5:
        score += support * 0.25
    if stats["capitalized_token_ratio"] > 0.4:
        score += support * 0.20
    if stats["punct_ratio"] > 0.10:
        score += support * 0.15
    if stats["edge_punct_count"] > 0:
        score += support * 0.10
    if role == "malicious" and stats["stopword_ratio"] > 0.6:
        score += support * 0.10
    return score


def select_signatures_for_llm_review(signatures: List[dict], *, role: str, max_review: int) -> List[dict]:
    """Select the highest-impact signatures for bounded LLM review."""
    if not signatures or max_review <= 0:
        return []

    supports = [float(sig.get("support", 0)) for sig in signatures]
    support_floor = _quantile(supports, 0.75, 0.0)
    scored = [(_review_score(sig, role, support_floor), sig) for sig in signatures]
    scored.sort(key=lambda item: (item[0], float(item[1].get("support", 0))), reverse=True)
    selected = [sig for _, sig in scored[:max_review]]

    log.info(
        "Selected %d/%d %s signatures for LLM review (support floor %.1f)",
        len(selected),
        len(signatures),
        role,
        support_floor,
    )
    return selected


def _build_prompt(role: str, batch: List[dict]) -> str:
    role_guidance = {
        "safe": (
            "Keep reusable benign-language patterns that generalize across prompts. "
            "Drop only if clearly a benchmark/task-format artifact, narrow named-entity/topic span, or overly specific template fragment."
        ),
        "malicious": (
            "Keep reusable attack or jailbreak intent patterns. "
            "Drop only if clearly an assistant response wrapper, persona boilerplate, dataset formatting token, or narrow topical fragment."
        ),
    }[role]

    lines = [
        "You are reviewing extracted signature candidates for a prompt-injection detector.",
        "Be conservative: if uncertain, keep the signature.",
        f"Role: {role.upper()}",
        role_guidance,
        'Return ONLY valid JSON in this format: {"drop_ids": [1, 2]}',
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
                    "other_support": other_support,
                    "precision": round(precision, 4),
                    "token_count": int(stats["token_count"]),
                    "alpha_ratio": round(stats["alpha_ratio"], 3),
                    "punct_ratio": round(stats["punct_ratio"], 3),
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
                "num_predict": 250,
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
    timeout_s: int = 45,
) -> List[dict]:
    """Apply a bounded LLM veto on a small statistically selected subset."""
    review_subset = select_signatures_for_llm_review(
        signatures,
        role=role,
        max_review=max_review,
    )
    if not review_subset:
        return signatures

    dropped_patterns: set[str] = set()
    for start in range(0, len(review_subset), batch_size):
        batch = review_subset[start:start + batch_size]
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
        "LLM filter dropped %d reviewed %s signatures; %d total remain",
        len(dropped_patterns),
        role,
        len(kept),
    )
    return kept