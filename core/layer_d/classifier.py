import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.LayerDResult import LayerDResult


@dataclass(frozen=True)
class Thresholds:
    low: float = 0.05
    high: float = 0.95

    def validate(self) -> None:
        if not (0.0 <= self.low <= 1.0 and 0.0 <= self.high <= 1.0):
            raise ValueError("Thresholds must be within [0,1]")
        if self.low >= self.high:
            raise ValueError("Expected low < high")


class LayerDClassifier:
    def __init__(
        self,
        model_dir: str,
        low: float = 0.05,
        high: float = 0.95,
        max_length: int = 512,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        self.thresholds = Thresholds(low=low, high=high)
        self.thresholds.validate()

    def _score_batch(self, texts):
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]

        return probs.float().detach().cpu().numpy()

    def predict(self, input_text) -> LayerDResult:
        start_time = time.time()
        probability_score = float(self._score_batch([input_text])[0])

        if probability_score < self.thresholds.low:
            verdict = "allow"
            confidence_score = 1.0 - probability_score
        elif probability_score < self.thresholds.high:
            verdict = "flag"
            confidence_score = 0.5
        else:
            verdict = "block"
            confidence_score = probability_score

        processing_time_ms = (time.time() - start_time) * 1000.0

        return LayerDResult(
            verdict=verdict,
            probability_score=probability_score,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
        )

    def predict_batch(self, texts):
        return self._score_batch(texts)
