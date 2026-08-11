"""Minimal SentenceTransformer-compatible ONNX encoder for the fast runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class OnnxSentenceEncoder:
    """Tokenize, pool, and normalize a local ONNX sentence encoder without Torch."""

    def __init__(self, model_dir: str | Path) -> None:
        import onnxruntime as ort  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.session = ort.InferenceSession(
            str(self.model_dir / "onnx" / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.pooling_mode = self._read_pooling_mode()

    def _read_pooling_mode(self) -> str:
        modules_path = self.model_dir / "modules.json"
        if modules_path.exists():
            modules = json.loads(modules_path.read_text(encoding="utf-8"))
            for module in modules:
                if "Pooling" not in str(module.get("type", "")):
                    continue
                config_path = self.model_dir / str(module.get("path", "")) / "config.json"
                if config_path.exists():
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    if config.get("pooling_mode_cls_token"):
                        return "cls"
                    if config.get("pooling_mode_mean_tokens"):
                        return "mean"
        return "mean"

    def encode(
        self,
        sentences: list[str] | str,
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **_: object,
    ) -> np.ndarray:
        del show_progress_bar, convert_to_numpy
        values = [sentences] if isinstance(sentences, str) else sentences
        encoded = self.tokenizer(
            values,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        inputs = {
            key: np.asarray(value, dtype=np.int64)
            for key, value in encoded.items()
            if key in self.input_names
        }
        output = np.asarray(self.session.run(None, inputs)[0], dtype=np.float32)
        if output.ndim == 3:
            if self.pooling_mode == "cls":
                output = output[:, 0, :]
            else:
                attention = np.asarray(encoded["attention_mask"], dtype=np.float32)[..., None]
                output = (output * attention).sum(axis=1) / np.maximum(attention.sum(axis=1), 1e-9)
        if normalize_embeddings:
            output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
        return output.astype(np.float32)
