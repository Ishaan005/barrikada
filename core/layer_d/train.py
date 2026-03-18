import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.layer_c.train.load_data import load_data
from core.layer_c.train.utils import write_json
from core.layer_d.train_eval import train_eval
from core.model_registry import current_git_commit, dir_sha256, file_sha256, update_latest_pointer, write_manifest
from core.settings import Settings


def main():
    settings = Settings()

    parser = argparse.ArgumentParser(description="Train Layer D ModernBERT classifier")
    parser.add_argument(
        "--csv",
        default=settings.dataset_path,
        help="Path to training CSV (columns: text,label)",
    )
    parser.add_argument(
        "--model-out",
        default=settings.layer_d_output_dir,
        help="Directory to write Hugging Face model artifacts",
    )
    parser.add_argument(
        "--report-out",
        default=settings.layer_d_report_path,
        help="Path to write evaluation report (.json)",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=settings.layer_d_low_threshold,
        help="Manual low routing threshold",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=settings.layer_d_high_threshold,
        help="Manual high routing threshold",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of Layer A+B filtered data",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Optional release version label (e.g. v2026.03.18-ld1)",
    )
    parser.add_argument(
        "--no-set-latest",
        action="store_true",
        help="Do not update releases/LATEST pointer when --model-version is used",
    )
    args = parser.parse_args()

    if args.low >= args.high:
        raise ValueError("Expected --low < --high")

    X, y, df = load_data(args.csv, use_cache=not args.no_cache)
    out = train_eval(X, y, model_out_dir=args.model_out, low=args.low, high=args.high)

    report_payload = {
        "model": out["model"],
        "thresholds": out["thresholds"],
        "metrics": out["metrics"],
        "model_info": out["model_info"],
        "hard_negative_mining": out.get("hard_negative_mining", {}),
    }
    write_json(args.report_out, report_payload)

    if args.model_version:
        release_root = Path(settings.layer_d_release_dir)
        release_dir = release_root / args.model_version
        release_model_dir = release_dir / "model"
        release_report_path = release_dir / "eval_report.json"

        source_model_dir = Path(out["model"]["artifact_dir"])
        release_dir.mkdir(parents=True, exist_ok=True)
        if release_model_dir.exists():
            shutil.rmtree(release_model_dir)
        shutil.copytree(source_model_dir, release_model_dir)
        write_json(str(release_report_path), report_payload)

        if not args.no_set_latest:
            update_latest_pointer(release_root, args.model_version)

        manifest = {
            "layer": "layer_d",
            "model_type": "modernbert_sequence_classifier",
            "model_version": args.model_version,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_git_commit(PROJECT_ROOT),
            "dataset": {
                "csv": args.csv,
                "rows_used": int(len(df)),
            },
            "thresholds": out["thresholds"],
            "artifacts": {
                "model_dir": str(release_model_dir.relative_to(PROJECT_ROOT)),
                "eval_report": str(release_report_path.relative_to(PROJECT_ROOT)),
            },
            "checksums": {
                "model_dir_sha256": dir_sha256(release_model_dir),
                "eval_report_sha256": file_sha256(release_report_path),
            },
        }
        manifest_path = release_dir / "manifest.json"
        write_manifest(manifest_path, manifest)
        print(f"Released Layer D model version: {args.model_version}")
        print(f"Release manifest: {manifest_path}")

    thresholds = out["thresholds"]
    metrics = out["metrics"]

    print("\n=== Layer D (ModernBERT) Evaluation ===")
    print(f"Routing thresholds: low={thresholds['low']:.4f}, high={thresholds['high']:.4f}")
    print("VAL (threshold=0.5):\n" + metrics["val"]["report_0.5"])
    print("VAL (routing thresholds):\n" + metrics["val"]["report_routing"])
    print(f"VAL verdict counts: {metrics['val']['routing_verdict_counts']}")
    print("\nTEST (threshold=0.5):\n" + metrics["test"]["report_0.5"])
    print("TEST (routing thresholds):\n" + metrics["test"]["report_routing"])
    print(f"TEST verdict counts: {metrics['test']['routing_verdict_counts']}")
    print(f"VAL ROC-AUC: {metrics['val']['roc_auc']:.4f}")
    print(f"TEST ROC-AUC: {metrics['test']['roc_auc']:.4f}")
    print(f"Rows used: {len(df)}")

if __name__ == "__main__":
    main()
