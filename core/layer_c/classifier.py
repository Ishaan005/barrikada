import time
from dataclasses import dataclass
from typing import Any

import torch
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

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
        model_path: str,
        embedding_model: str = "all-mpnet-base-v2",
        low: float = 0.35,
        high: float = 0.85,
    ):
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(embedding_model, device=_device)
        artifact = joblib.load(model_path)

        self.model = artifact.get("model")
        self.calibrator = artifact.get("calibrator")

        if self.model is None or not hasattr(self.model, "predict_proba"):
            raise ValueError("Layer C model artifact does not contain a valid predict_proba model")

        self.thresholds = Thresholds(low=low, high=high)
        self.thresholds.validate()

    def predict(self, input_text) -> LayerCResult:
        start_time = time.time()

        emb = self.encoder.encode([input_text], normalize_embeddings=True)
        probability_score = float(self.model.predict_proba(emb)[:, 1][0])
        if self.calibrator is not None:
            probability_score = float(self.calibrator.predict(np.array([probability_score]))[0])

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
        )

    def predict_dict(self, input_text):
        """Something to get a simple dict output for API responses."""
        res = self.predict(input_text)
        return {"score": res.probability_score, "decision": res.verdict}

    def predict_batch(self, texts):
        """Return raw probability scores for a batch of texts."""
        embs = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        probs = self.model.predict_proba(embs)[:, 1]
        if self.calibrator is not None:
            probs = self.calibrator.predict(probs)
        return probs