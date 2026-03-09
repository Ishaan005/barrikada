import time
import hashlib
from typing import List, Tuple, Optional
from pathlib import Path
import yara

from core.settings import Settings
from models.SignatureMatch import SignatureMatch, Severity
from models.LayerBResult import LayerBResult


class SignatureEngine:
    def __init__(self):
        self.settings = Settings()
        self.malicious_rules: Optional[yara.Rules] = None
        self.allow_rules: Optional[yara.Rules] = None
        self._load_signatures()

    def _load_signatures(self):
        extracted_high = Path(self.settings.layer_b_malicious_rules_path)
        extracted_allow = Path(self.settings.layer_b_allow_rules_path)

        if not extracted_high.exists():
            raise FileNotFoundError(
                "Extracted MALICIOUS signatures not found. Run scripts/extract_signature_patterns.py "
                f"to generate {extracted_high}"
            )

        self.malicious_rules = yara.compile(filepath=str(extracted_high))
        print(f"Loaded extracted MALICIOUS signatures: {extracted_high.name}")

        if extracted_allow.exists():
            self.allow_rules = yara.compile(filepath=str(extracted_allow))
            print(f"Loaded extracted SAFE allow signatures: {extracted_allow.name}")
        else:
            self.allow_rules = None
            print(f"SAFE allow signatures not found ({extracted_allow.name}); allowlisting disabled.")

    def _is_allowlisted(self, text: str) -> Tuple[bool, List[str]]:
        """Return (is_allowlisted, matched_rule_ids) for the given text."""
        if self.allow_rules is None:
            return False, []
        try:
            matches = self.allow_rules.match(data=text)
            rule_ids = [m.rule for m in matches]
            return bool(rule_ids), rule_ids
        except Exception as e:
            print(f"YARA allowlisting error: {e}")
            return False, []

    def _match_malicious_yara(self, text: str) -> List[SignatureMatch]:
        """Match extracted malicious YARA rules against text, returning per-instance matches."""
        if self.malicious_rules is None:
            return []

        matches: List[SignatureMatch] = []
        try:
            for match in self.malicious_rules.match(data=text):
                precision_raw = match.meta.get("precision")
                try:
                    precision = float(precision_raw) if precision_raw is not None else 1.0
                except (TypeError, ValueError):
                    precision = 1.0

                tags = list(getattr(match, "tags", None) or [])
                rule_type = match.meta.get("type")
                if rule_type:
                    tags.append(str(rule_type))

                # Prefer the original extracted pattern text for interpretability
                description = match.meta.get("pattern", match.rule)

                for string_match in match.strings:
                    for instance in string_match.instances:
                        matched_data = instance.matched_data.decode("utf-8", errors="ignore")
                        matches.append(SignatureMatch(
                            rule_id=match.rule,
                            severity=Severity.MALICIOUS,
                            pattern=string_match.identifier,
                            matched_text=matched_data,
                            start_pos=instance.offset,
                            end_pos=instance.offset + len(matched_data),
                            rule_description=description,
                            tags=tags,
                            confidence=precision,
                        ))
        except Exception as e:
            print(f"YARA matching error: {e}")

        return matches

    def detect(self, text: str) -> LayerBResult:
        start_time = time.time()
        input_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Check for malicious patterns first — prevents evasion by padding with safe text.
        all_matches = self._match_malicious_yara(text)

        if all_matches:
            verdict, confidence = self._calculate_verdict(all_matches)
            return LayerBResult(
                input_hash=input_hash,
                processing_time_ms=(time.time() - start_time) * 1000,
                matches=all_matches,
                verdict=verdict,
                confidence_score=confidence,
                allowlisted=False,
                allowlist_rules=[],
            )

        # No malicious matches — check allow-list for early exit.
        allowlisted, allow_rules = self._is_allowlisted(text)
        if allowlisted:
            return LayerBResult(
                input_hash=input_hash,
                processing_time_ms=(time.time() - start_time) * 1000,
                matches=[],
                verdict="allow",
                confidence_score=self.settings.layer_b_allow_confidence,
                allowlisted=True,
                allowlist_rules=allow_rules,
            )

        # Not allowlisted, no matches — verdict depends on setting.
        no_match_verdict = self.settings.layer_b_no_match_verdict
        no_match_confidence = self.settings.layer_b_no_match_confidence
        return LayerBResult(
            input_hash=input_hash,
            processing_time_ms=(time.time() - start_time) * 1000,
            matches=[],
            verdict=no_match_verdict,
            confidence_score=no_match_confidence,
            allowlisted=False,
            allowlist_rules=[],
        )

    def _calculate_verdict(self, matches: List[SignatureMatch]) -> Tuple[str, float]:
        """Return (verdict, confidence) for a non-empty list of malicious signature matches.

        Blocking logic (two-tier):
          1. Primary block: >= N distinct high-precision word-boundary ($re) matches.
          2. Secondary block: >= M high-precision $re matches AND >= K distinct $s matches
             (char n-gram corroboration lowers the $re threshold).

        Allow logic:
          If there are *no* $re hits at all and only weak $s evidence (below a confidence
          threshold), the input is allowed — the $s hits are likely coincidental n-gram
          overlaps rather than genuine malice.

        Everything else is flagged for downstream layers.
        """
        # Dedup per-instance matches, keeping highest confidence per description.
        word_hits: dict[str, float] = {}   # $re matches
        s_hits: dict[str, float] = {}      # $s matches
        for m in matches:
            if m.pattern == "$re":
                word_hits[m.rule_description] = max(
                    word_hits.get(m.rule_description, 0.0), m.confidence
                )
            elif m.pattern == "$s":
                s_hits[m.rule_description] = max(
                    s_hits.get(m.rule_description, 0.0), m.confidence
                )

        qualifying_re = sum(
            1 for conf in word_hits.values()
            if conf >= self.settings.layer_b_block_min_rule_precision
        )
        unique_s = len(s_hits)

        # --- Primary block: strong word-boundary evidence ---
        if qualifying_re >= self.settings.layer_b_block_min_hits:
            return "block", self.settings.layer_b_block_confidence

        # --- Secondary block: moderate $re + corroborating $s ---
        if (qualifying_re >= self.settings.layer_b_block_secondary_re_hits
                and unique_s >= self.settings.layer_b_block_secondary_s_hits):
            return "block", self.settings.layer_b_block_confidence * 0.9

        # --- Allow: very weak evidence (only $s, no $re, low confidence) ---
        if len(word_hits) == 0:
            s_conf_sum = sum(s_hits.values())
            if s_conf_sum < self.settings.layer_b_allow_max_s_confidence:
                return "allow", self.settings.layer_b_no_match_confidence

        return "flag", self.settings.layer_b_flag_confidence