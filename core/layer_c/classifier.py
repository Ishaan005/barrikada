"""Layer C (Tier-2) classifier inference wrapper.

This layer is only called for *unsure* cases after upstream rule/signature tiers.
It must be fast, auditable, and production-viable.

Model:
- TF-IDF word + character n-grams
- Logistic Regression (class_weight='balanced')

Routing:
- score < T_low  => allow
- T_low <= score < T_high => flag (review)
- score >= T_high => block
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import joblib
from models.LayerCResult import LayerCResult


@dataclass(frozen=True)
class Thresholds:
    """Deployment thresholds for routing decisions."""

    low: float = 0.35
    high: float = 0.85

    def validate(self) -> None:
        if not (0.0 <= self.low <= 1.0 and 0.0 <= self.high <= 1.0):
            raise ValueError("Thresholds must be within [0,1]")
        if self.low >= self.high:
            raise ValueError("Expected low < high")


class Classifier:
    """Tier-2 classifier used by the orchestrator.

    This class loads persisted artifacts and exposes a single-text `predict()`
    interface that returns a standardized LayerCResult.
    """

    def __init__(
        self,
        vectorizer_path: str,
        model_path: str,
        low: float = 0.35,
        high: float = 0.85,
        model_version: str = "tf_idf_logreg_v1",
    ):
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)

        self.thresholds = Thresholds(low=low, high=high)
        self.thresholds.validate()
        self.model_version = model_version

    def predict(self, input_text: str) -> LayerCResult:
        """Score a single input string.

        Returns:
            LayerCResult with:
            - probability_score: P(prompt_injection=1)
            - verdict: allow | flag | block (thresholded)
        """

        start_time = time.time()

        X = self.vectorizer.transform([input_text])
        probability_score = self.model.predict_proba(X)[:, 1][0]

        if probability_score < self.thresholds.low:
            verdict = "allow"
        elif probability_score < self.thresholds.high:
            verdict = "flag"
        else:
            verdict = "block"

        # Confidence: distance from the decision boundary (simple, monotonic)
        if verdict == "allow":
            confidence_score = 1.0 - probability_score
        elif verdict == "block":
            confidence_score = probability_score
        else:
            # Middle band = inherently uncertain
            confidence_score = 0.5

        processing_time_ms = (time.time() - start_time) * 1000.0

        return LayerCResult(
            verdict=verdict,
            probability_score=probability_score,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            model_version=self.model_version,
        )

    def predict_dict(self, input_text: str) -> Dict[str, float | str]:
        """Something to get a simple dict output for API responses."""
        res = self.predict(input_text)
        return {"score": res.probability_score, "decision": res.verdict}

