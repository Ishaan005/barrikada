import argparse
import sys
from pathlib import Path

# Allow running as a script: `python core/layer_c/train.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.settings import Settings
from core.layer_c.train.load_data import load_data
from core.layer_c.train.utils import save, write_json
from core.layer_c.train.train_eval import train_eval

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

    args = parser.parse_args()

    #train only on samples that would reach Layer C.
    X, y, df = load_data(args.csv, use_cache=not args.no_cache)
    out = train_eval(X, y, low=args.low, high=args.high)

    artifact = out["artifact"]
    thresholds = out["thresholds"]
    metrics = out["metrics"]
    emb_info = out["embedding_info"]
    feature_importance_top = out["feature_importance_top"]

    save(artifact, args.model_out)
    write_json(args.report_out, {
        "thresholds": thresholds,
        "metrics": metrics,
        "embedding_info": emb_info,
        "feature_importance_top": feature_importance_top,
        "model_metadata": artifact.get("metadata", {}),
    })

    #console output
    print("\n=== Layer C Evaluation ===")
    print(
        "Routing thresholds: "
        f"low={thresholds['low']:.4f}, high={thresholds['high']:.4f} "
    )
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
