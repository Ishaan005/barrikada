import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import joblib
import numpy as np
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
        reducer_path: Optional[str] = None,
        low= 0.35,
        high= 0.85,
    ):
        self.vectorizer = joblib.load(vectorizer_path)
        self.reducer = joblib.load(reducer_path) if reducer_path else None
        self.model = joblib.load(model_path)

        self.thresholds = Thresholds(low=low, high=high)
        self.thresholds.validate()

    def predict(self, input_text: str) -> LayerCResult:
        start_time = time.time()

        X = self.vectorizer.transform([input_text])
        if self.reducer is not None:
            X = self.reducer.transform(X)
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
        )

    def predict_dict(self, input_text):
        """Something to get a simple dict output for API responses."""
        res = self.predict(input_text)
        return {"score": res.probability_score, "decision": res.verdict}

    def grid_search(
        self,
        texts: List[str],
        labels: List[int],
        low_range: Tuple[float, float, float] = (0.1, 0.5, 0.05),
        high_range: Tuple[float, float, float] = (0.5, 0.95, 0.05),
        optimize_for: str = "f1",
    ):
        # Pre-compute all probabilities once
        X_vec = self.vectorizer.transform(texts)
        if self.reducer is not None:
            X_vec = self.reducer.transform(X_vec)
        probs = self.model.predict_proba(X_vec)[:, 1]
        labels = np.array(labels) #type:ignore
        
        low_vals = np.arange(*low_range)
        high_vals = np.arange(*high_range)
        
        best_score = -1.0
        best_thresholds = (0.35, 0.85)
        best_metrics = {}
        
        for low in low_vals:
            for high in high_vals:
                if low >= high:
                    continue
                    
                # Predict: flag/block (prob >= low) counts as detection
                preds = (probs >= low).astype(int)
                
                tp = ((preds == 1) & (labels == 1)).sum()
                fp = ((preds == 1) & (labels == 0)).sum()
                fn = ((preds == 0) & (labels == 1)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                score = {"f1": f1, "recall": recall, "precision": precision}[optimize_for]
                
                if score > best_score:
                    best_score = score
                    best_thresholds = (low, high)
                    best_metrics = {"precision": precision, "recall": recall, "f1": f1}
        
        return {
            "best_low": best_thresholds[0],
            "best_high": best_thresholds[1],
            "optimized_for": optimize_for,
            **best_metrics,
        }