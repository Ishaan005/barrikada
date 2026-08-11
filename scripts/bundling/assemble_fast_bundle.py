"""Assemble the minimal Layer B-D artifact tree used by the fast service image."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FILE_MAPPINGS = {
    "core/layer_b/signatures/embeddings/faiss_index.bin": "layer_b/embeddings/faiss_index.bin",
    "core/layer_b/signatures/embeddings/centroids.npy": "layer_b/embeddings/centroids.npy",
    "core/layer_b/signatures/embeddings/benign_faiss_index.bin": "layer_b/embeddings/benign_faiss_index.bin",
    "core/layer_b/signatures/embeddings/benign_centroids.npy": "layer_b/embeddings/benign_centroids.npy",
    "core/layer_b/signatures/embeddings/cluster_radii.json": "layer_b/embeddings/cluster_radii.json",
    "core/layer_b/signatures/embeddings/metadata.json": "layer_b/embeddings/metadata.json",
    "core/layer_c/outputs/classifier.onnx": "layer_c/classifier.onnx",
    "core/layer_c/outputs/calibrator.joblib": "layer_c/calibrator.joblib",
}

DIRECTORY_MAPPINGS = {
    "core/layer_b/signatures/embeddings/prompt_encoder_onnx": "layer_b/embeddings/prompt_encoder_onnx",
    "core/layer_c/outputs/encoder_onnx": "layer_c/encoder_onnx",
    "core/layer_d/outputs/onnx": "layer_d/onnx",
}


def assemble(
    source_root: Path, destination: Path, parity_dir: Path, layer_d_onnx_dir: Path | None = None
) -> None:
    source_root = source_root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for source_name, destination_name in FILE_MAPPINGS.items():
        source = source_root / source_name
        if not source.is_file():
            raise SystemExit(f"fast-profile source artifact is missing: {source_name}")
        target = destination / destination_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source_name, destination_name in DIRECTORY_MAPPINGS.items():
        source = (
            layer_d_onnx_dir
            if destination_name == "layer_d/onnx" and layer_d_onnx_dir is not None
            else source_root / source_name
        )
        if not source.is_dir():
            raise SystemExit(f"fast-profile source directory is missing: {source_name}")
        shutil.copytree(source, destination / destination_name, dirs_exist_ok=True)

    for layer in ("layer_b", "layer_c", "layer_d"):
        report = parity_dir / f"{layer}.json"
        if not report.is_file():
            raise SystemExit(f"parity report is missing: {report}")
        parity_target = destination / "parity" / report.name
        parity_target.parent.mkdir(parents=True, exist_ok=True)
        if report.resolve() != parity_target.resolve():
            shutil.copy2(report, parity_target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--parity-dir", type=Path, required=True)
    parser.add_argument("--layer-d-onnx-dir", type=Path)
    args = parser.parse_args()
    assemble(args.source_root, args.destination, args.parity_dir, args.layer_d_onnx_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
