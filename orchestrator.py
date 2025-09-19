#Pipeline orchestrator

import time
import hashlib
from typing import Dict, List, Any, Tuple

from models.PipelineResult import PipelineResult
    
class PIPipeline():
    def __init__(self, enable_early_termination: bool = True):

        self.enable_early_termination = enable_early_termination
        
        # Init Layer A 
        from core.layer_a.pipeline import analyze_text
        self.layer_a_analyze = analyze_text

        # Init Layer B
        from core.layer_b.signature_engine import SignatureEngine
        self.layer_b_engine = SignatureEngine()
    
    #Run complete detection pipeline on input text
    def detect(self, input_text: str) -> PipelineResult:

        start_time = time.time()
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]
        
        # Convert string to bytes for Layer A
        input_bytes = input_text.encode('utf-8')
        
        # Layer A: Text preprocessing and basic detection
        layer_a_start = time.time()

        layer_a_result = self.layer_a_analyze(input_bytes)

        layer_a_time = (time.time() - layer_a_start) * 1000
        
        #TODO: Check for early termination after Layer A
        if self.enable_early_termination and layer_a_result.get('suspicious', False):
            high_confidence_flags = ['direction_override', 'embedded_encodings']
            if any(flag in layer_a_result.get('flags', []) for flag in high_confidence_flags):
                # High confidence detection, might skip Layer B
                pass  # For now, continue to Layer B for completeness
        
        # Layer B: Signature-based detection
        layer_b_start = time.time()

        # Use the cleaned text from Layer 
        analysis_text = layer_a_result.get('final', input_text)
        layer_b_detection = self.layer_b_engine.detect(analysis_text)
        layer_b_result = layer_b_detection.to_dict()

        layer_b_time = (time.time() - layer_b_start) * 1000
        
        # Aggregate results and make final decision
        final_verdict, confidence_score, risk_score, detected_threats, recommended_action = self._aggregate_results(
            layer_a_result, layer_b_result
        )
        
        total_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            input_hash=input_hash,
            total_processing_time_ms=total_time,
            layer_a_result=layer_a_result,
            layer_a_time_ms=layer_a_time,
            layer_b_result=layer_b_result,
            layer_b_time_ms=layer_b_time,
            final_verdict=final_verdict,
            confidence_score=confidence_score,
            risk_score=risk_score,
            detected_threats=detected_threats,
            recommended_action=recommended_action
        )
    
    # Aggregate results from all layers into final verdict
    def _aggregate_results(self, layer_a_result: Dict[str, Any], layer_b_result: Dict[str, Any]) -> Tuple[str, float, float, List[str], str]:
        detected_threats = []
        risk_factors = []
        
        # Analyze Layer A results
        layer_a_suspicious = layer_a_result.get('suspicious', False)
        layer_a_flags = layer_a_result.get('flags', [])
        
        if layer_a_suspicious:
            for flag in layer_a_flags:
                detected_threats.extend([f"layer_a_{flag}"])
            
            # Weight Layer A flags by severity (adjusted for better accuracy)
            high_risk_flags = ['direction_override', 'embedded_encodings']
            medium_risk_flags = ['confusable_chars']
            low_risk_flags = ['possible_base64']
            
            for flag in layer_a_flags:
                if flag in high_risk_flags:
                    risk_factors.append(30.0)  # High contribution to risk
                elif flag in medium_risk_flags:
                    risk_factors.append(15.0)  # Medium contribution
                elif flag in low_risk_flags:
                    risk_factors.append(8.0)   # Reduced low contribution
                else:
                    risk_factors.append(5.0)   # Low contribution
        
        # Analyze Layer B results
        layer_b_verdict = layer_b_result.get('verdict', 'allow')
        layer_b_matches = layer_b_result.get('matches', [])
        layer_b_score = layer_b_result.get('total_score', 0.0)
        layer_b_severity = layer_b_result.get('highest_severity')
        
        if layer_b_matches:
            detected_threats.extend([f"signature_{match['rule_id']}" for match in layer_b_matches])
            
            # Add Layer B risk contribution
            if layer_b_severity == 'high':
                risk_factors.append(50.0)  # Very high risk
            elif layer_b_severity == 'medium':
                risk_factors.append(25.0)  # High risk
            elif layer_b_severity == 'low':
                risk_factors.append(10.0)  # Moderate risk
        
        # Calculate aggregated risk score (0-100)
        risk_score = min(100.0, sum(risk_factors))
        
        # Determine final verdict based on highest severity findings
        if layer_b_verdict == 'block' or any(flag in ['direction_override'] for flag in layer_a_flags):
            # Only block on the most serious threats
            final_verdict = 'block'
            confidence_score = 0.72  # Reduced from 0.9 to be more conservative
            recommended_action = "Block immediately. High confidence threat detected"
            
        elif (layer_b_verdict == 'flag' or 
              any(flag in ['embedded_encodings', 'confusable_chars'] for flag in layer_a_flags) or 
              risk_score > 25):  # Increased threshold from 30 to 25
            final_verdict = 'flag'
            confidence_score = 0.56  # Reduced from 0.7 to match accuracy target
            recommended_action = "Flag. Suspicious patterns detected"
            
        else:
            final_verdict = 'allow'
            confidence_score = 0.96  # High confidence in clean content
            recommended_action = "Allow. No significant threats detected"
        
        # Adjust confidence based on agreement between layers
        layer_agreement = self._calculate_layer_agreement(layer_a_result, layer_b_result)
        confidence_score = min(1.0, confidence_score * layer_agreement)
        
        return final_verdict, confidence_score, risk_score, detected_threats, recommended_action
    
    def _calculate_layer_agreement(self, layer_a_result: Dict[str, Any], layer_b_result: Dict[str, Any]) -> float:
        """
        Calculate agreement between layers (used to adjust confidence)
        
        Returns:
            Agreement factor (0.5 to 1.5) - higher means more agreement
        """
        layer_a_suspicious = layer_a_result.get('suspicious', False)
        layer_b_suspicious = layer_b_result.get('verdict', 'allow') != 'allow'
        
        if layer_a_suspicious == layer_b_suspicious:
            return 1.2  # Good agreement boosts confidence
        else:
            return 0.8  # Disagreement reduces confidence