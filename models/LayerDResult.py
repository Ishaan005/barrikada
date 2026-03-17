from dataclasses import dataclass

@dataclass
class LayerDResult:
    """Standardized result from Layer D (ModernBERT classifier)."""

    verdict: str
    probability_score: float
    confidence_score: float
    processing_time_ms: float

    def to_dict(self):
        return {
            "verdict": self.verdict,
            "probability_score": self.probability_score,
            "confidence_score": self.confidence_score,
            "processing_time_ms": self.processing_time_ms,
        }
