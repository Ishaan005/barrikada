"""Pipeline orchestrator (decision cascade).

Flow:
1) Layer A preprocesses text and may hard-block for high-confidence flags.
2) Layer B runs extracted YARA rules and SAFE allowlisting.
    - If MALICIOUS (block) => final verdict = block
    - If SAFE (allowlisted) => final verdict = allow
    - If unsure (flag) => send to Layer C
3) Layer C makes the final decision only for unsure cases.

Each layer makes its own decision; we do not aggregate scores.
"""

import hashlib
import time

from core.settings import Settings
from models.PipelineResult import PipelineResult

from models.verdicts import DecisionLayer, FinalVerdict


class PIPipeline:
    def __init__(self):

        from core.layer_a.pipeline import analyze_text
        from core.layer_b.signature_engine import SignatureEngine
        from core.layer_c.classifier import Classifier

        self.layer_a_analyze = analyze_text
        self.layer_b_engine = SignatureEngine()

        settings = Settings()
        self.layer_c_classifier = Classifier(
            vectorizer_path=settings.vectorizer_path,
            model_path=settings.model_path,
            low=settings.layer_c_low_threshold,
            high=settings.layer_c_high_threshold,
        )

    def detect(self, input_text: str) -> PipelineResult:
        start_time = time.time()
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]

        # ----- Layer A -----
        layer_a_result = self.layer_a_analyze(input_text)
        analysis_text = layer_a_result.processed_text

        # Hard-block from Layer A (high-confidence flags)
        if layer_a_result.get_verdict() == "block":
            total_time = (time.time() - start_time) * 1000
            return PipelineResult(
                input_hash=input_hash,
                total_processing_time_ms=total_time,
                layer_a_result=layer_a_result.to_dict(),
                layer_a_time_ms=layer_a_result.processing_time_ms,
                layer_b_result=None,
                layer_b_time_ms=None,
                layer_c_result=None,
                layer_c_time_ms=None,
                final_verdict=FinalVerdict.BLOCK,
                decision_layer=DecisionLayer.LAYER_A,
                confidence_score=layer_a_result.confidence_score,
            )

        # ----- Layer B -----
        layer_b_result = self.layer_b_engine.detect(analysis_text)

        # SAFE allowlisting => allow immediately (optimization)
        if (not layer_a_result.suspicious) and getattr(layer_b_result, "allowlisted", False):
            total_time = (time.time() - start_time) * 1000
            return PipelineResult(
                input_hash=input_hash,
                total_processing_time_ms=total_time,
                layer_a_result=layer_a_result.to_dict(),
                layer_a_time_ms=layer_a_result.processing_time_ms,
                layer_b_result=layer_b_result.to_dict(),
                layer_b_time_ms=layer_b_result.processing_time_ms,
                layer_c_result=None,
                layer_c_time_ms=None,
                final_verdict=FinalVerdict.ALLOW,
                decision_layer=DecisionLayer.LAYER_B,
                confidence_score=layer_b_result.confidence_score,
            )

        # MALICIOUS signatures => block immediately
        if layer_b_result.verdict == "block":
            total_time = (time.time() - start_time) * 1000
            return PipelineResult(
                input_hash=input_hash,
                total_processing_time_ms=total_time,
                layer_a_result=layer_a_result.to_dict(),
                layer_a_time_ms=layer_a_result.processing_time_ms,
                layer_b_result=layer_b_result.to_dict(),
                layer_b_time_ms=layer_b_result.processing_time_ms,
                layer_c_result=None,
                layer_c_time_ms=None,
                final_verdict=FinalVerdict.BLOCK,
                decision_layer=DecisionLayer.LAYER_B,
                confidence_score=layer_b_result.confidence_score,
            )

        # ----- Layer C -----
        # For security: anything not allowlisted SAFE is screened by Layer C.
        # This prevents malicious prompts with no signature hits from being allowed.
        layer_c_result = self.layer_c_classifier.predict(analysis_text)

        total_time = (time.time() - start_time) * 1000
        return PipelineResult(
            input_hash=input_hash,
            total_processing_time_ms=total_time,
            layer_a_result=layer_a_result.to_dict(),
            layer_a_time_ms=layer_a_result.processing_time_ms,
            layer_b_result=layer_b_result.to_dict(),
            layer_b_time_ms=layer_b_result.processing_time_ms,
            layer_c_result=layer_c_result.to_dict(),
            layer_c_time_ms=layer_c_result.processing_time_ms,
            final_verdict=FinalVerdict(layer_c_result.verdict),
            decision_layer=DecisionLayer.LAYER_C,
            confidence_score=layer_c_result.confidence_score,
        )

    
