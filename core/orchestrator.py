"""Pipeline orchestrator (decision cascade).

Flow:
1) Layer A preprocesses text and may hard-block for high-confidence flags.
2) Layer B runs extracted YARA rules and SAFE allowlisting.
    - If MALICIOUS (block) => final verdict = block
    - If SAFE (allowlisted) => final verdict = allow
    - If unsure (flag) => send to Layer C
3) Layer C makes the final decision only for unsure cases.
    - If block or allow => final verdict
    - If flag (uncertain) => escalate to Layer E
4) Layer E (LLM judge) is the final arbiter for cases that remain
   uncertain after all prior layers.

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
        from core.layer_e.llm_judge import LLMJudge

        settings = Settings()

        self.layer_a_analyze = analyze_text
        self.layer_b_engine = SignatureEngine()
        self.layer_c_classifier = Classifier(
            vectorizer_path=settings.vectorizer_path,
            model_path=settings.model_path,
            reducer_path=settings.reducer_path,
            low=settings.layer_c_low_threshold,
            high=settings.layer_c_high_threshold,
        )
        self.layer_e_judge = LLMJudge()

    def _create_result(
        self,
        input_hash,
        start_time,
        layer_a_result,
        final_verdict: FinalVerdict,
        decision_layer: DecisionLayer,
        confidence_score: float,
        layer_b_result=None,
        layer_c_result=None,
        layer_e_result_dict=None,
        layer_e_time_ms=None,
    ):
        total_time = (time.time() - start_time) * 1000
        return PipelineResult(
            input_hash=input_hash,
            total_processing_time_ms=total_time,
            layer_a_result=layer_a_result.to_dict(),
            layer_a_time_ms=layer_a_result.processing_time_ms,
            layer_b_result=layer_b_result.to_dict() if layer_b_result else None,
            layer_b_time_ms=layer_b_result.processing_time_ms if layer_b_result else None,
            layer_c_result=layer_c_result.to_dict() if layer_c_result else None,
            layer_c_time_ms=layer_c_result.processing_time_ms if layer_c_result else None,
            layer_e_result=layer_e_result_dict,
            layer_e_time_ms=layer_e_time_ms,
            final_verdict=final_verdict,
            decision_layer=decision_layer,
            confidence_score=confidence_score,
        )

    def detect(self, input_text: str) -> PipelineResult:
        start_time = time.time()
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]

        # ----- Layer A -----
        layer_a_result = self.layer_a_analyze(input_text)
        analysis_text = layer_a_result.processed_text

        # Hard-block from Layer A (high-confidence flags)
        if layer_a_result.get_verdict() == "block":
            return self._create_result(
                input_hash, start_time, layer_a_result,
                final_verdict=FinalVerdict.BLOCK,
                decision_layer=DecisionLayer.LAYER_A,
                confidence_score=layer_a_result.confidence_score,
            )

        # ----- Layer B -----
        layer_b_result = self.layer_b_engine.detect(analysis_text)

        # SAFE allowlisting => allow immediately (optimization)
        if (not layer_a_result.suspicious) and getattr(layer_b_result, "allowlisted", False):
            return self._create_result(
                input_hash, start_time, layer_a_result,
                layer_b_result=layer_b_result,
                final_verdict=FinalVerdict.ALLOW,
                decision_layer=DecisionLayer.LAYER_B,
                confidence_score=layer_b_result.confidence_score,
            )

        # MALICIOUS signatures => block immediately
        if layer_b_result.verdict == "block":
            return self._create_result(
                input_hash, start_time, layer_a_result,
                layer_b_result=layer_b_result,
                final_verdict=FinalVerdict.BLOCK,
                decision_layer=DecisionLayer.LAYER_B,
                confidence_score=layer_b_result.confidence_score,
            )

        # ----- Layer C -----
        # For security: anything not allowlisted SAFE is screened by Layer C.
        # This prevents malicious prompts with no signature hits from being allowed.
        layer_c_result = self.layer_c_classifier.predict(analysis_text)

        if layer_c_result.verdict == "block" or layer_c_result.verdict == "allow":
            return self._create_result(
                input_hash, start_time, layer_a_result,
                layer_b_result=layer_b_result,
                layer_c_result=layer_c_result,
                final_verdict=FinalVerdict(layer_c_result.verdict),
                decision_layer=DecisionLayer.LAYER_C,
                confidence_score=layer_c_result.confidence_score,
            )
    
        # ----- Layer E -----
        layer_e_start = time.time()
        layer_e_result = self.layer_e_judge.call_judge(analysis_text)
        layer_e_time_ms = (time.time() - layer_e_start) * 1000

        # Fallback if LLM judge returns None unexpectedly
        if layer_e_result is None:
            layer_e_result_dict = {"label": 1, "rationale": "LLM judge returned None"}
            layer_e_verdict = FinalVerdict.BLOCK
        else:
            layer_e_result_dict = {
                "label": layer_e_result.label,
                "rationale": layer_e_result.rationale,
            }
            # Map JudgeOutput label (0=benign, 1=malicious) to FinalVerdict
            layer_e_verdict = FinalVerdict.BLOCK if layer_e_result.label == 1 else FinalVerdict.ALLOW

        return self._create_result(
            input_hash, start_time, layer_a_result,
            layer_b_result=layer_b_result,
            layer_c_result=layer_c_result,
            layer_e_result_dict=layer_e_result_dict,
            layer_e_time_ms=layer_e_time_ms,
            final_verdict=layer_e_verdict,
            decision_layer=DecisionLayer.LAYER_E,
            confidence_score=1.0,  # LLM judge gives binary decisions
        )

    
