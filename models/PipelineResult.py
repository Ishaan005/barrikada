from dataclasses import dataclass
from typing import Dict, Any, List

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
    
    # Final aggregated results
    final_verdict: str | None # "allow", "flag", "block" 
    confidence_score: float | None  # 0.0 to 1.0
    risk_score: float | None  # 0 to 100
    detected_threats: List[str] | None
    recommended_action: str | None

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
            'final_verdict': self.final_verdict,
            'confidence_score': self.confidence_score,
            'risk_score': self.risk_score,
            'detected_threats': self.detected_threats,
            'recommended_action': self.recommended_action
        }