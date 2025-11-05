#Pipeline orchestrator

import time
import hashlib

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

        #Init Layer C - ML Classifier
        from core.layer_c.classifier import Classifier
        self.layer_c_classifier = Classifier(
            vectorizer_path="../models/tf_idf_vectorizer.joblib",
            model_path="../models/tf_idf_logreg.joblib",
            low=0.25,
            high=0.75
        )
    
    #Run complete detection pipeline on input text
    def detect(self, input_text: str) -> PipelineResult:
        """
        Run complete detection pipeline on input text
        
        Args:
            input_text: Text to analyze (string)
            
        Returns:
            PipelineResult: Complete analysis results from all layers
        """
        start_time = time.time()
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]
        
        ##### Layer A: Text preprocessing and basic detection #####
        # Layer A now accepts both bytes and strings
        layer_a_result = self.layer_a_analyze(input_text)
        
        # Check for early termination after Layer A
        if self.enable_early_termination and layer_a_result.suspicious:
            high_confidence_flags = ['direction_override', 'embedded_encodings']
            if any(flag in layer_a_result.flags for flag in high_confidence_flags):
                # High confidence detection, might skip Layer B
                pass  # For now, continue to Layer B for completeness
        
        ##### Layer B: Signature-based detection ######
        # Use the cleaned text from Layer A
        analysis_text = layer_a_result.processed_text
        layer_b_result = self.layer_b_engine.detect(analysis_text)

        ##### Layer C: ML-based classification #####
        layer_c_result = self.layer_c_classifier.predict(analysis_text)
        
        # Aggregate results and make final decision
        final_verdict, confidence_score, risk_score, detected_threats, recommended_action = self._aggregate_results(
            layer_a_result, layer_b_result, layer_c_result
        )
        
        total_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            input_hash=input_hash,
            total_processing_time_ms=total_time,
            layer_a_result=layer_a_result.to_dict(),
            layer_a_time_ms=layer_a_result.processing_time_ms,
            layer_b_result=layer_b_result.to_dict(),
            layer_b_time_ms=layer_b_result.processing_time_ms,
            final_verdict=final_verdict,
            confidence_score=confidence_score,
            risk_score=risk_score,
            detected_threats=detected_threats,
            recommended_action=recommended_action
        )
    
    # Aggregate results from all layers into final verdict
    def _aggregate_results(self, layer_a_result, layer_b_result, layer_c_result):
        """
        Aggregate results from all layers using standardized result objects
        
        Args:
            layer_a_result: LayerAResult object
            layer_b_result: LayerBResult object
            layer_c_result: LayerCResult object
            
        Returns:
            Tuple of (verdict, confidence, risk_score, threats, action)
        """
        detected_threats = []
        risk_factors = []
        
        # Analyze Layer A results
        if layer_a_result.suspicious:
            for flag in layer_a_result.flags:
                detected_threats.append(f"layer_a_{flag}")
            
            # Add Layer A risk contribution
            risk_factors.append(layer_a_result.get_risk_score())
        
        # Analyze Layer B results
        if layer_b_result.matches:
            detected_threats.extend([f"signature_{match.rule_id}" for match in layer_b_result.matches])
            
            # Add Layer B risk contribution
            risk_factors.append(layer_b_result.get_risk_score())
        
        # Analyze Layer C results
        if layer_c_result.verdict != 'allow':
            detected_threats.append(f"ml_classifier_{layer_c_result.verdict}")
            
            # Add Layer C risk contribution
            risk_factors.append(layer_c_result.get_risk_score())
        
        # Calculate aggregated risk score (0-100)
        risk_score = min(100.0, sum(risk_factors))
        
        # Determine final verdict based on highest severity findings
        # Collect all verdicts
        verdicts = [layer_a_result.get_verdict(), layer_b_result.verdict, layer_c_result.verdict]
        
        # Block if any layer says block
        if 'block' in verdicts:
            final_verdict = 'block'
            # Average confidence from layers that detected threats
            confidences = []
            if layer_a_result.suspicious:
                confidences.append(layer_a_result.confidence_score)
            if layer_b_result.matches:
                confidences.append(layer_b_result.confidence_score)
            if layer_c_result.verdict == 'block':
                confidences.append(layer_c_result.confidence_score)
            confidence_score = sum(confidences) / len(confidences) if confidences else 0.7
            recommended_action = "Block immediately. High confidence threat detected"
            
        # Flag if any layer says flag (and none say block)
        elif 'flag' in verdicts:
            final_verdict = 'flag'
            confidences = []
            if layer_a_result.suspicious:
                confidences.append(layer_a_result.confidence_score)
            if layer_b_result.matches:
                confidences.append(layer_b_result.confidence_score)
            if layer_c_result.verdict == 'flag':
                confidences.append(layer_c_result.confidence_score)
            confidence_score = sum(confidences) / len(confidences) if confidences else 0.6
            recommended_action = "Flag for review. Suspicious patterns detected"
            
        else:
            final_verdict = 'allow'
            # High confidence in clean content when all layers agree
            confidence_score = 0.95
            recommended_action = "Allow. No significant threats detected"
        
        # Adjust confidence based on agreement between layers
        layer_agreement = self._calculate_layer_agreement(layer_a_result, layer_b_result, layer_c_result)
        confidence_score = min(1.0, confidence_score * layer_agreement)
        
        return final_verdict, confidence_score, risk_score, detected_threats, recommended_action
    
    def _calculate_layer_agreement(self, layer_a_result, layer_b_result, layer_c_result):
        """
        Calculate agreement between layers (used to adjust confidence)
        
        Args:
            layer_a_result: LayerAResult object
            layer_b_result: LayerBResult object
            layer_c_result: LayerCResult object
        
        Returns:
            Agreement factor (0.8 to 1.2) - higher means more agreement
        """
        # Get verdicts from each layer
        verdicts = [
            layer_a_result.get_verdict(),
            layer_b_result.verdict,
            layer_c_result.verdict
        ]
        
        # Count how many layers agree on the most common verdict
        from collections import Counter
        verdict_counts = Counter(verdicts)
        most_common_verdict, count = verdict_counts.most_common(1)[0]
        
        # Perfect agreement (all 3 layers)
        if count == 3:
            return 1.2
        # Majority agreement (2 out of 3)
        elif count == 2:
            return 1.0
        # No agreement (all different)
        else:
            return 0.8