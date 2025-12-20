import time
import hashlib
from typing import List, Tuple, Optional
from pathlib import Path
import yara

from models.SignatureMatch import SignatureMatch, Severity
from models.LayerBResult import LayerBResult

class SignatureEngine:
    
    def __init__(self):
        self.signatures_dir = Path(__file__).parent / "signatures" / "extracted"
        self.malicious_rules: Optional[yara.Rules] = None
        self.allow_rules: Optional[yara.Rules] = None
        self._load_signatures()
    
    def _load_signatures(self):
        extracted_high = self.signatures_dir / "malicious_block_high_signatures.yar"
        extracted_allow = self.signatures_dir / "safe_allow_signatures.yar"

        # Ensure extracted HIGH signatures exist. Adding this here for open source collaborators.
        if not extracted_high.exists():
            raise FileNotFoundError(
                "Extracted HIGH signatures not found. Run scripts/extract_signature_patterns.py "
                "to generate core/layer_b/signatures/extracted/malicious_block_high_signatures.yar"
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
                tags = match.meta.get("tags", "").split() if match.meta.get("tags") else []
                description = match.meta.get("description", match.rule)

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
                                confidence=1.0,
                            )
                        )
        except Exception as e:
            print(f"YARA matching error: {e}")

        return matches
    
    def detect(self, text: str) -> LayerBResult:
        start_time = time.time()
        input_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        allowlisted, allow_rules = self._is_allowlisted(text)
        if allowlisted:
            return LayerBResult(
                input_hash=input_hash,
                processing_time_ms=(time.time() - start_time) * 1000,
                matches=[],
                verdict="allow",
                confidence_score=0.99,
                allowlisted=True,
                allowlist_rules=allow_rules,
            )

        all_matches = self._match_malicious_yara(text)
        
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
            return "allow", 1.0

        # Extracted signatures are high-precision indicators.
        # Treat a single hit as suspicious, multiple hits as high confidence.
        if len(matches) >= 2:
            verdict, confidence = "block", 0.95
        else:
            verdict, confidence = "flag", 0.85

        return verdict, confidence