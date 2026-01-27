import time
from dataclasses import dataclass

import joblib
from models.LayerCResult import LayerCResult


@dataclass(frozen=True)
class Thresholds:
    low: float = 0.35
    high: float = 0.85

    def validate(self) -> None:
        if not (0.0 <= self.low <= 1.0 and 0.0 <= self.high <= 1.0):
            raise ValueError("Thresholds must be within [0,1]")
        if self.low >= self.high:
            raise ValueError("Expected low < high")


class Classifier:
    def __init__(
        self,
        vectorizer_path: str,
        model_path: str,
        low= 0.35,
        high= 0.85,
        model_version: str = "tf_idf_logreg_v1",
    ):
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)

        self.thresholds = Thresholds(low=low, high=high)
        self.thresholds.validate()
        self.model_version = model_version

    def predict(self, input_text: str) -> LayerCResult:
        start_time = time.time()

        X = self.vectorizer.transform([input_text])
        probability_score = self.model.predict_proba(X)[:, 1][0]

        if probability_score < self.thresholds.low:
            verdict = "allow"
        elif probability_score < self.thresholds.high:
            verdict = "flag"
        else:
            verdict = "block"

        # Confidence: distance from the decision boundary
        if verdict == "allow":
            confidence_score = 1.0 - probability_score
        elif verdict == "block":
            confidence_score = probability_score
        else:
            # Middle band = uncertain
            confidence_score = 0.5

        processing_time_ms = (time.time() - start_time) * 1000.0

        return LayerCResult(
            verdict=verdict,
            probability_score=probability_score,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            model_version=self.model_version,
        )

    def predict_dict(self, input_text):
        """Something to get a simple dict output for API responses."""
        res = self.predict(input_text)
        return {"score": res.probability_score, "decision": res.verdict}

