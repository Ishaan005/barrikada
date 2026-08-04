from dataclasses import dataclass
from typing import Any

from .SignatureMatch import Severity, SignatureMatch


@dataclass
class DetectionResult:
    """Complete detection result from Layer B"""

    input_hash: str
    processing_time_ms: float
    matches: list[SignatureMatch]
    verdict: str  # "allow", "flag", "block"
    total_score: float
    highest_severity: Severity | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "input_hash": self.input_hash,
            "processing_time_ms": self.processing_time_ms,
            "matches": [
                {
                    "rule_id": match.rule_id,
                    "severity": match.severity.value,
                    "pattern": match.pattern,
                    "matched_text": match.matched_text,
                    "start_pos": match.start_pos,
                    "end_pos": match.end_pos,
                    "rule_description": match.rule_description,
                    "tags": match.tags,
                    "confidence": match.confidence,
                }
                for match in self.matches
            ],
            "verdict": self.verdict,
            "total_score": self.total_score,
            "highest_severity": self.highest_severity.value if self.highest_severity else None,
        }
