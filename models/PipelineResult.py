from dataclasses import dataclass
from typing import Any

from models.LayerResult import LayerResult
from models.verdicts import DecisionLayer, FinalVerdict


def _serialize_result(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return result


@dataclass
class PipelineResult:
    input_hash: str
    total_processing_time_ms: float

    # Layer A results
    layer_a_result: dict[str, Any] | LayerResult
    layer_a_time_ms: float

    # Layer B results
    layer_b_result: dict[str, Any] | LayerResult | None
    layer_b_time_ms: float | None

    # Layer C results
    layer_c_result: dict[str, Any] | LayerResult | None
    layer_c_time_ms: float | None

    # Layer D results
    layer_d_result: dict[str, Any] | LayerResult | None
    layer_d_time_ms: float | None

    # Layer E results (LLM judge)
    layer_e_result: dict[str, Any] | LayerResult | None
    layer_e_time_ms: float | None

    # Final decision (decision cascade)
    final_verdict: FinalVerdict
    decision_layer: DecisionLayer  # "A", "B", "C", "D", or "E"
    confidence_score: float  # confidence of the deciding layer

    def to_dict(self) -> dict[str, Any]:
        # Convert to dictionary for outpput
        return {
            "input_hash": self.input_hash,
            "total_processing_time_ms": self.total_processing_time_ms,
            "layer_a_result": _serialize_result(self.layer_a_result),
            "layer_a_time_ms": self.layer_a_time_ms,
            "layer_b_result": _serialize_result(self.layer_b_result),
            "layer_b_time_ms": self.layer_b_time_ms,
            "layer_c_result": _serialize_result(self.layer_c_result),
            "layer_c_time_ms": self.layer_c_time_ms,
            "layer_d_result": _serialize_result(self.layer_d_result),
            "layer_d_time_ms": self.layer_d_time_ms,
            "layer_e_result": _serialize_result(self.layer_e_result),
            "layer_e_time_ms": self.layer_e_time_ms,
            "final_verdict": self.final_verdict.value,
            "decision_layer": self.decision_layer.value,
            "confidence_score": self.confidence_score,
        }
