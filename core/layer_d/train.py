import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.layer_c.train.load_data import load_data
from core.layer_c.train.utils import write_json
from core.layer_d.train_eval import train_eval
from core.settings import Settings


def main() -> int:
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
    args = parser.parse_args()

    if args.low >= args.high:
        raise ValueError("Expected --low < --high")

    X, y, df = load_data(args.csv, use_cache=not args.no_cache)
    out = train_eval(X, y, model_out_dir=args.model_out, low=args.low, high=args.high)

    write_json(
        args.report_out,
        {
            "model": out["model"],
            "thresholds": out["thresholds"],
            "metrics": out["metrics"],
            "model_info": out["model_info"],
        },
    )

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
