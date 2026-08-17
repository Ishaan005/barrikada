"""Create a CPU-oriented INT8 Layer D ONNX bundle from the FP32 export."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=PROJECT_ROOT / "core/layer_d/outputs/onnx",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=PROJECT_ROOT / "core/layer_d/outputs/onnx-int8",
    )
    args = parser.parse_args()
    source_model = args.src / "model.onnx"
    if not source_model.is_file():
        raise SystemExit(f"Layer D ONNX source is missing: {source_model}")
    args.dst.mkdir(parents=True, exist_ok=True)
    destination_model = args.dst / "model.onnx"
    quantize_dynamic(
        model_input=str(source_model),
        model_output=str(destination_model),
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        source = args.src / name
        if not source.is_file():
            raise SystemExit(f"Layer D runtime metadata is missing: {source}")
        shutil.copy2(source, args.dst / name)
    calibration = args.src / "jentic-reviewed-calibration.json"
    if calibration.is_file():
        shutil.copy2(calibration, args.dst / calibration.name)
    if destination_model.stat().st_size >= source_model.stat().st_size:
        raise SystemExit("quantized Layer D model did not reduce artifact size")
    print(
        f"Layer D INT8 model: {source_model.stat().st_size / 1024**2:.1f} MiB -> "
        f"{destination_model.stat().st_size / 1024**2:.1f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
