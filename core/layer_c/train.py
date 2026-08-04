import argparse
from datetime import datetime, timezone
from pathlib import Path

from core.layer_c.train.load_data import load_data
from core.layer_c.train.train_eval import train_eval
from core.layer_c.train.utils import save, write_json
from core.settings import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    settings = Settings()

    parser = argparse.ArgumentParser(description="Train Layer-C Tier-2 classifier")
    parser.add_argument(
        "--csv",
        default=settings.dataset_path,
        help="Path to training CSV (columns: text,label)",
    )
    parser.add_argument(
        "--model-out",
        default=settings.model_path,
        help="Path to write model artifact (.joblib)",
    )
    parser.add_argument(
        "--report-out",
        default=str(Path("test_results") / "layer_c_eval_latest.json"),
        help="Path to write evaluation report (.json)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of Layer A+B filtered data",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=settings.layer_c_low_threshold,
        help="Manual low threshold for allow/flag boundary",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=settings.layer_c_high_threshold,
        help="Manual high threshold for flag/block boundary",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Optional release version label (e.g. v2026.03.18-lc1)",
    )
    parser.add_argument(
        "--no-set-latest",
        action="store_true",
        help="Do not update releases/LATEST pointer when --model-version is used",
    )

    args = parser.parse_args()

    # train only on samples that would reach Layer C.
    X, y, df = load_data(args.csv, use_cache=not args.no_cache)
    out = train_eval(
        X,
        y,
        low=args.low,
        high=args.high,
    )

    artifact = out["artifact"]
    thresholds = out["thresholds"]
    metrics = out["metrics"]
    emb_info = out["embedding_info"]
    feature_importance_top = out["feature_importance_top"]

    save(artifact, args.model_out)
    report_payload = {
        "thresholds": thresholds,
        "metrics": metrics,
        "embedding_info": emb_info,
        "feature_importance_top": feature_importance_top,
        "model_metadata": artifact.get("metadata", {}),
    }
    write_json(args.report_out, report_payload)

    if args.model_version:
        release_root = Path(settings.layer_c_release_dir)
        release_dir = release_root / args.model_version
        release_dir.mkdir(parents=True, exist_ok=True)

        release_model_path = release_dir / "classifier.joblib"
        save(artifact, str(release_model_path))

        release_report_path = release_dir / "eval_report.json"
        write_json(str(release_report_path), report_payload)

        if not args.no_set_latest:
            print("Skipping update of releases/LATEST pointer (model_registry removed)")

        manifest = {
            "layer": "layer_c",
            "model_type": "xgboost_embedding_classifier",
            "model_version": args.model_version,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": None,
            "dataset": {
                "csv": args.csv,
                "rows_used": int(len(df)),
            },
            "thresholds": thresholds,
            "artifacts": {
                "classifier": str(release_model_path.relative_to(PROJECT_ROOT)),
                "eval_report": str(release_report_path.relative_to(PROJECT_ROOT)),
            },
            "checksums": {
                "classifier_sha256": None,
                "eval_report_sha256": None,
            },
        }
        manifest_path = release_dir / "manifest.json"
        write_json(str(manifest_path), manifest)
        print(f"Released Layer C model version: {args.model_version}")
        print(f"Release manifest: {manifest_path}")

    # console output
    print("\n=== Layer C Evaluation ===")
    print(f"Routing thresholds: low={thresholds['low']:.4f}, high={thresholds['high']:.4f} ")
    print(f"Threshold source: {thresholds.get('source', 'manual')}")
    print("VAL (threshold=0.5, calibrated):\n" + metrics["val"]["calibrated"]["report_0.5"])
    print("VAL (routing thresholds):\n" + metrics["val"]["report_routing"])
    print(f"VAL verdict counts: {metrics['val']['routing_verdict_counts']}")
    print("\nTEST (threshold=0.5, calibrated):\n" + metrics["test"]["calibrated"]["report_0.5"])
    print("TEST (routing thresholds):\n" + metrics["test"]["report_routing"])
    print(f"TEST verdict counts: {metrics['test']['routing_verdict_counts']}")
    print(f"VAL ROC-AUC (raw): {metrics['val']['raw']['roc_auc']:.4f}")
    print(f"VAL ROC-AUC (calibrated): {metrics['val']['calibrated']['roc_auc']:.4f}")
    print(f"TEST ROC-AUC (raw): {metrics['test']['raw']['roc_auc']:.4f}")
    print(f"TEST ROC-AUC (calibrated): {metrics['test']['calibrated']['roc_auc']:.4f}")
    print(f"VAL PR-AUC (calibrated): {metrics['val']['calibrated']['pr_auc']:.4f}")
    print(f"TEST PR-AUC (calibrated): {metrics['test']['calibrated']['pr_auc']:.4f}")
    print(f"Rows used: {len(df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
