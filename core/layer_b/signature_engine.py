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
        self.signatures_dir = Path(self.settings.layer_b_signatures_extracted_dir)
        self.malicious_rules: Optional[yara.Rules] = None
        self.allow_rules: Optional[yara.Rules] = None
        self._load_signatures()
    
    def _load_signatures(self):
        extracted_high = Path(self.settings.layer_b_malicious_rules_path)
        extracted_allow = Path(self.settings.layer_b_allow_rules_path)

        # Ensure extracted HIGH signatures exist. Adding this here for open source collaborators.
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
            # Allowlisting is an optimization; pipeline remains correct without it.
            self.allow_rules = None
            print(f"SAFE allow signatures not found ({extracted_allow.name}); "
                    "allowlisting disabled."
            )


    def _is_allowlisted(self, text: str) -> Tuple[bool, List[str]]:
        """Return whether SAFE allow rules matched, and which rule IDs matched."""
        if self.allow_rules is None:
            return False, []
        try:
            matches = self.allow_rules.match(data=text)
            rule_ids = [m.rule for m in matches]
            return (len(rule_ids) > 0), rule_ids
        except Exception as e:
            print(f"YARA allowlisting error: {e}")
            return False, []


    def _match_malicious_yara(self, text: str) -> List[SignatureMatch]:
        """Match extracted malicious YARA rules.

        Note: extracted signatures are treated as MALICIOUS indicators.
        """
        if self.malicious_rules is None:
            return []

        matches: List[SignatureMatch] = []
        try:
            yara_matches = self.malicious_rules.match(data=text)
            for match in yara_matches:
                # Extracted rules include useful meta fields (pattern, precision, support).
                # YARA does not support float meta values, so our generator stores them as strings.
                precision_raw = match.meta.get("precision", None)
                try:
                    precision = float(precision_raw) if precision_raw is not None else 1.0
                except Exception:
                    precision = 1.0

                tags = list(getattr(match, "tags", []) or [])
                rule_type = match.meta.get("type", None)
                if rule_type:
                    tags.append(str(rule_type))

                # Prefer the original extracted pattern for interpretability.
                description = match.meta.get("pattern", match.rule)

                for string_match in match.strings:
                    for instance in string_match.instances:
                        matched_data = instance.matched_data.decode("utf-8", errors="ignore")
                        matches.append(
                            SignatureMatch(
                                rule_id=match.rule,
                                severity=Severity.MALICIOUS,
                                pattern=string_match.identifier,
                                matched_text=matched_data,
                                start_pos=instance.offset,
                                end_pos=instance.offset + len(matched_data),
                                rule_description=description,
                                tags=tags,
                                confidence=precision,
                            )
                        )
        except Exception as e:
            print(f"YARA matching error: {e}")

        return matches
    
    def detect(self, text: str) -> LayerBResult:
        start_time = time.time()
        input_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Always check for malicious patterns first (prevents evasion by padding with safe text)
        all_matches = self._match_malicious_yara(text)
        
        # If malicious patterns found, verdict based on those (ignore allowlist)
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
        
        # No malicious patterns - check if allowlisted for early exit
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
        
        # No malicious matches, not allowlisted - flag for ML review
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
    
    def _calculate_verdict(self, matches: List[SignatureMatch]) -> Tuple[str, float]:
        if not matches:
            # No signature matches and not allowlisted - flag for ML classifier review
            return "flag", self.settings.layer_b_flag_confidence

        # IMPORTANT: `matches` contains per-instance matches. A single YARA rule can match
        # multiple times in one prompt, which can incorrectly inflate match counts.
        # Use distinct matched RULE IDs for verdicting.
        # Collapse per-instance matches down to distinct rules and their (max) confidence.
        # Our extractor stores rule precision in YARA meta and we copy it into
        # SignatureMatch.confidence.
        rule_to_conf: dict[str, float] = {}
        for m in matches:
            prev = rule_to_conf.get(m.rule_id)
            rule_to_conf[m.rule_id] = m.confidence if prev is None else max(prev, m.confidence)

        distinct_rule_hits = len(rule_to_conf)
        min_rule_precision = min(rule_to_conf.values()) if rule_to_conf else 0.0

        # Extremely conservative hard-block policy:
        # - require multiple distinct rules
        # - require those rules to be very high precision
        # Otherwise: FLAG for review / ML escalation.
        if (
            distinct_rule_hits >= self.settings.layer_b_block_min_hits
            and min_rule_precision >= self.settings.layer_b_block_min_rule_precision
        ):
            verdict, confidence = "block", self.settings.layer_b_block_confidence
        else:
            verdict, confidence = "flag", self.settings.layer_b_flag_confidence

        return verdict, confidence