"""Small training script for Layer C (Tier-2 ML).

Baseline: TF-IDF (word + char) + Logistic Regression.
Saves artifacts + prints metrics + writes top features.
"""

import argparse
import sys
from pathlib import Path

# Allow running as a script: `python core/layer_c/train.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.settings import Settings
from core.layer_c.training_utils import load_data, save, train_eval, write_json

def main():
    settings = Settings()

    parser = argparse.ArgumentParser(description="Train Layer-C Tier-2 classifier")
    parser.add_argument(
        "--csv",
        default=settings.dataset_path,
        help="Path to training CSV (columns: text,label)",
    )
    parser.add_argument(
        "--vectorizer-out",
        default=settings.vectorizer_path,
        help="Path to write vectorizer artifact (.joblib)",
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

    args = parser.parse_args()

    # Default behavior: train only on samples that would reach Layer C.
    X, y, df = load_data(args.csv)
    out = train_eval(X, y)

    vec = out["vectorizer"]
    model = out["model"]
    thresholds = out["thresholds"]
    metrics = out["metrics"]

    save(vec, model, args.vectorizer_out, args.model_out)
    write_json(args.report_out, {"thresholds": thresholds, "metrics": metrics})

    # Console output for quick feedback
    print("\n=== Layer C (Tier-2) Evaluation ===")
    tuning = thresholds.get("tuning")
    print(
        "Routing thresholds: "
        f"low={thresholds['low']:.4f}, high={thresholds['high']:.4f} "
    )
    print(
            "VAL policy metrics: "
            f"block_precision={tuning.get('val_block_precision')}, "
            f"block_recall={tuning.get('val_block_recall')}, "
            f"malicious_allow_rate={tuning.get('val_malicious_allow_rate')}, "
            f"allow_rate={tuning.get('val_allow_rate')}"
    )
    print("VAL (threshold=0.5):\n" + metrics["val"]["report_0.5"])
    print("VAL (routing thresholds):\n" + metrics["val"]["report_routing"])
    print(f"VAL verdict counts: {metrics['val']['routing_verdict_counts']}")
    print("\nTEST (threshold=0.5):\n" + metrics["test"]["report_0.5"])
    print("TEST (routing thresholds):\n" + metrics["test"]["report_routing"])
    print(f"TEST verdict counts: {metrics['test']['routing_verdict_counts']}")
    print(f"VAL ROC-AUC: {metrics['val']['roc_auc']:.4f}")
    print(f"TEST ROC-AUC: {metrics['test']['roc_auc']:.4f}")
    print(f"Rows used: {len(df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
