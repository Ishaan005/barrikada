from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .SignatureMatch import SignatureMatch, Severity

@dataclass
class LayerBResult:
    """Standardized result from Layer B (Signature Detection)"""
    
    # Detection results
    matches: List[SignatureMatch]
    verdict: str  # "allow", "flag", "block"
    total_score: float
    highest_severity: Optional[Severity]
    confidence_score: float  # 0.0 to 1.0 - confidence in detection
    
    # Processing metadata
    processing_time_ms: float
    input_hash: str

    # Allow-listing metadata (used for early termination / skipping later layers)
    allowlisted: bool = False
    allowlist_rules: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'input_hash': self.input_hash,
            'processing_time_ms': self.processing_time_ms,
            'matches': [
                {
                    'rule_id': match.rule_id,
                    'severity': match.severity.value,
                    'pattern': match.pattern,
                    'matched_text': match.matched_text,
                    'start_pos': match.start_pos,
                    'end_pos': match.end_pos,
                    'rule_description': match.rule_description,
                    'tags': match.tags,
                    'confidence': match.confidence
                }
                for match in self.matches
            ],
            'verdict': self.verdict,
            'total_score': self.total_score,
            'highest_severity': self.highest_severity.value if self.highest_severity else None,
            'confidence_score': self.confidence_score,
            'allowlisted': self.allowlisted,
            'allowlist_rules': list(self.allowlist_rules),
        }
    
    def get_risk_score(self) -> float:
        """Calculate risk score contribution (0-100)"""
        if not self.matches:
            return 0.0
        
        severity_weights = {
            Severity.HIGH: 50.0,
            Severity.MEDIUM: 25.0,
            Severity.LOW: 10.0
        }
        
        # Base risk on highest severity
        base_risk = severity_weights.get(self.highest_severity, 0.0) if self.highest_severity else 0.0
        
        # Add incremental risk for multiple matches
        match_count_bonus = min(20.0, len(self.matches) * 5.0)
        
        return min(100.0, base_risk + match_count_bonus)
