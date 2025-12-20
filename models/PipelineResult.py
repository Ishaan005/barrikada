from dataclasses import dataclass
from typing import Any, Dict

from models.verdicts import DecisionLayer, FinalVerdict

@dataclass
class PipelineResult:
    input_hash: str
    total_processing_time_ms: float
    
    # Layer A results
    layer_a_result: Dict[str, Any]
    layer_a_time_ms: float

    # Layer B results
    layer_b_result: Dict[str, Any] | None
    layer_b_time_ms: float | None

    # Layer C results
    layer_c_result: Dict[str, Any] | None
    layer_c_time_ms: float | None
    
    # Final decision (decision cascade)
    final_verdict: FinalVerdict
    decision_layer: DecisionLayer  # "A", "B", or "C"
    confidence_score: float  # confidence of the deciding layer

    def to_dict(self) -> Dict[str, Any]:
        #Convert to dictionary for outpput
        return {
            'input_hash': self.input_hash,
            'total_processing_time_ms': self.total_processing_time_ms,
            'layer_a_result': self.layer_a_result,
            'layer_a_time_ms': self.layer_a_time_ms,
            'layer_b_result': self.layer_b_result,
            'layer_b_time_ms': self.layer_b_time_ms,
            'layer_c_result': self.layer_c_result,
            'layer_c_time_ms': self.layer_c_time_ms,
            'final_verdict': self.final_verdict.value,
            'decision_layer': self.decision_layer.value,
            'confidence_score': self.confidence_score,
        }