"""Evaluate PT/reference and ONNX fast backends on a deterministic labeled sample.

This command writes the three reports required by the signed fast-runtime bundle.
Reports are evidence, not overrides: a backend that disagrees, regresses false
blocks/recall, or fails to improve latency remains failed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[4]
INTEGRATION_ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, INTEGRATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.layer_b.signature_engine import SignatureEngine  # noqa: E402
from core.layer_c.classifier import Classifier  # noqa: E402
from core.layer_d.classifier import LayerDClassifier  # noqa: E402
from core.onnx_parity import Verdict, evaluate_backend_parity  # noqa: E402
from core.settings import Settings  # noqa: E402

from scripts.evaluation.jentic_corpus import load_corpus  # noqa: E402


def _load_balanced_sample(path: Path, per_class: int) -> tuple[list[str], list[str]]:
    by_label: dict[str, list[str]] = {"0": [], "1": []}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("label", "")).strip()
            text = row.get("text")
            if label in by_label and isinstance(text, str) and len(by_label[label]) < per_class:
                by_label[label].append(text)
            if all(len(values) >= per_class for values in by_label.values()):
                break
    if any(len(values) < per_class for values in by_label.values()):
        raise SystemExit(f"dataset does not contain {per_class} rows for both labels: {path}")
    texts: list[str] = []
    labels: list[str] = []
    for index in range(per_class):
        for label in ("0", "1"):
            texts.append(by_label[label][index])
            labels.append("benign" if label == "0" else "attack")
    return texts, labels


def _predict(
    detector: Callable[[str], Verdict], texts: Sequence[str]
) -> tuple[list[Verdict], float]:
    detector(texts[0])
    verdicts: list[Verdict] = []
    durations: list[float] = []
    for text in texts:
        started = time.perf_counter()
        verdicts.append(detector(text))
        durations.append((time.perf_counter() - started) * 1000.0)
    return verdicts, statistics.quantiles(durations, n=100, method="inclusive")[94]


def _write_report(
    output_dir: Path,
    layer: str,
    reference: tuple[list[Verdict], float],
    candidate: tuple[list[Verdict], float],
    labels: Sequence[str],
) -> dict:
    report = evaluate_backend_parity(
        reference[0],
        candidate[0],
        labels,  # type: ignore[arg-type]
        reference_p95_ms=reference[1],
        candidate_p95_ms=candidate[1],
    )
    report.update(
        {
            "layer": layer,
            "reference_backend": "pytorch",
            "candidate_backend": "onnxruntime-cpu",
            "dataset_digest_scope": "text-and-labels-not-persisted",
            "evaluation_environment": {
                "machine": platform.machine(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{layer}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _layer_b(
    texts: Sequence[str],
) -> tuple[tuple[list[Verdict], float], tuple[list[Verdict], float]]:
    engine = SignatureEngine()
    candidate_model = engine.model
    reference_model = SentenceTransformer(
        str(ROOT / "core/layer_b/signatures/embeddings/prompt_encoder"), device="cpu"
    )
    engine.model = reference_model
    reference = _predict(lambda text: engine.detect(text).verdict, texts)
    engine.model = candidate_model
    del reference_model
    gc.collect()
    candidate = _predict(lambda text: engine.detect(text).verdict, texts)
    return reference, candidate


def _cached_mpnet_snapshot() -> Path:
    cache_root = (
        Path.home()
        / ".cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots"
    )
    snapshots = sorted(path for path in cache_root.glob("*") if path.is_dir())
    if not snapshots:
        raise SystemExit("Layer C reference encoder is not present in the Hugging Face cache")
    return snapshots[-1]


def _layer_c(
    texts: Sequence[str],
) -> tuple[tuple[list[Verdict], float], tuple[list[Verdict], float]]:
    outputs = ROOT / "core/layer_c/outputs"
    with tempfile.TemporaryDirectory(prefix="barrikade-layer-c-reference-") as directory:
        anchor = Path(directory) / "classifier.joblib"
        shutil.copy2(outputs / "classifier.joblib", anchor)
        reference_model = Classifier(
            model_path=anchor,
            embedding_model=str(_cached_mpnet_snapshot()),
            low=Settings().layer_c_low_threshold,
            high=Settings().layer_c_high_threshold,
        )
        reference = _predict(lambda text, model=reference_model: model.predict(text).verdict, texts)
        del reference_model
        gc.collect()

    candidate_model = Classifier(
        model_path=outputs / "classifier.joblib",
        low=Settings().layer_c_low_threshold,
        high=Settings().layer_c_high_threshold,
    )
    candidate = _predict(lambda text: candidate_model.predict(text).verdict, texts)
    return reference, candidate


def _layer_d(
    texts: Sequence[str],
    candidate_dir: Path,
    reference_dir: Path,
    low: float,
    high: float,
    max_length: int,
) -> tuple[tuple[list[Verdict], float], tuple[list[Verdict], float]]:
    reference_model = LayerDClassifier(
        model_dir=reference_dir,
        low=low,
        high=high,
        max_length=max_length,
    )
    reference = _predict(lambda text, model=reference_model: model.predict(text).verdict, texts)
    del reference_model
    gc.collect()

    with tempfile.TemporaryDirectory(prefix="barrikade-layer-d-candidate-") as directory:
        candidate_root = Path(directory)
        (candidate_root / "model").symlink_to(reference_dir.resolve(), target_is_directory=True)
        (candidate_root / "onnx").symlink_to(candidate_dir.resolve(), target_is_directory=True)
        candidate_model = LayerDClassifier(
            model_dir=candidate_root / "model",
            low=low,
            high=high,
            max_length=max_length,
        )
        candidate = _predict(lambda text: candidate_model.predict(text).verdict, texts)
    return reference, candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/barrikade_test.csv")
    parser.add_argument("--jentic-corpus", type=Path)
    parser.add_argument("--jentic-split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument(
        "--layer-d-candidate-dir",
        type=Path,
        default=ROOT / "core/layer_d/outputs/onnx",
    )
    parser.add_argument(
        "--layer-d-reference-dir",
        type=Path,
        default=ROOT / "core/layer_d/outputs/model",
    )
    parser.add_argument("--layer-d-low", type=float, default=Settings().layer_d_low_threshold)
    parser.add_argument("--layer-d-high", type=float, default=Settings().layer_d_high_threshold)
    parser.add_argument("--layer-d-max-length", type=int, default=512)
    parser.add_argument(
        "--layers", nargs="+", choices=("layer_b", "layer_c", "layer_d"), default=None
    )
    args = parser.parse_args()
    if args.samples_per_class < 2:
        raise SystemExit("--samples-per-class must be at least 2")

    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")

    if args.jentic_corpus:
        cases = [
            case for case in load_corpus(args.jentic_corpus) if case.split == args.jentic_split
        ]
        if not cases:
            raise SystemExit(f"Jentic corpus contains no {args.jentic_split!r} cases")
        texts = [case.text for case in cases]
        labels = ["benign" if case.label == "benign" else "attack" for case in cases]
    else:
        texts, labels = _load_balanced_sample(args.dataset, args.samples_per_class)
    evaluators = {
        "layer_b": _layer_b,
        "layer_c": _layer_c,
        "layer_d": lambda values: _layer_d(
            values,
            args.layer_d_candidate_dir,
            args.layer_d_reference_dir,
            args.layer_d_low,
            args.layer_d_high,
            args.layer_d_max_length,
        ),
    }
    selected = args.layers or list(evaluators)
    failed = False
    for layer in selected:
        print(f"Evaluating {layer} on {len(texts)} balanced samples...", flush=True)
        report = _write_report(args.output_dir, layer, *evaluators[layer](texts), labels)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        failed = failed or report["passed"] is not True
        gc.collect()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
