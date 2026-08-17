"""
Download the smallest Qwen3Guard model bundle into core/models/layer_e/.

Usage:
    python scripts/bundling/download_qwen3guard.py
    python scripts/bundling/download_qwen3guard.py --force
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from core.settings import Settings, env_value


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Qwen3Guard-Gen-0.6B into the Layer E model bundle directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the existing local bundle if it already exists",
    )
    args = parser.parse_args()

    settings = Settings()
    destination = Path(
        env_value(
            "BARRIKADE_LAYER_E_MODEL_DIR",
            str(PROJECT_ROOT / "core" / "models" / "layer_e" / "qwen3guard-barrikade"),
        )
    )

    if destination.exists() and any(destination.iterdir()):
        if not args.force:
            print(f"Layer E bundle already exists at {destination}")
            print("Use --force to re-download it from Hugging Face.")
            return 0
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(  # nosec B615
        repo_id=settings.layer_e_model_hf_id,
        local_dir=str(destination),
    )  # type: ignore
    print(f"Downloaded {settings.layer_e_model_hf_id} to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
