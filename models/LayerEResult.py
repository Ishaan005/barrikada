from dataclasses import dataclass
from typing import Any

from models.LayerResult import LayerResult


@dataclass
class LayerEResult(LayerResult):
    """Standardized result from Layer E (LLM Judge)"""

    # Detection results
    verdict: str  # "allow", "block"
    rationale: str
    model: str
    no_think: bool
    raw_response: str

    # Metadata/Tokens
    processing_time_ms: float
    confidence_score: float = 1.0
    reasoning_trace: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "verdict": self.verdict,
            "rationale": self.rationale,
            "model": self.model,
            "no_think": self.no_think,
            "raw_response": self.raw_response,
            "processing_time_ms": self.processing_time_ms,
            "confidence_score": self.confidence_score,
            "reasoning_trace": self.reasoning_trace,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def get_risk_score(self) -> float:
        """Calculate risk score contribution (0-100)"""
        return 100.0 if self.verdict == "block" else 0.0
