import argparse

from core.layer_c.train.load_data import load_data
from core.layer_c.train.utils import write_json
from core.layer_e.train_eval import train_teacher_qwen35
from core.settings import Settings


def main():
    settings = Settings()

    parser = argparse.ArgumentParser(description="Train Layer E Qwen3.5 teacher with QLoRA")
    parser.add_argument(
        "--csv", default=settings.dataset_path, help="Path to CSV with text,label columns"
    )
    parser.add_argument(
        "--model-out",
        default=settings.layer_e_teacher_output_dir,
        help="Directory to write merged teacher artifacts",
    )
    parser.add_argument(
        "--report-out",
        default=settings.layer_e_teacher_report_path,
        help="Path to write evaluation report (.json)",
    )
    parser.add_argument(
        "--teacher-model-id",
        default=settings.layer_e_teacher_hf_model_id,
        help="Hugging Face model id for Qwen3.5 4B",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable Layer A+B filtering cache")
    args = parser.parse_args()

    X, y, _ = load_data(args.csv, use_cache=not args.no_cache)
    out = train_teacher_qwen35(
        X,
        y,
        output_dir=args.model_out,
        teacher_model_id=args.teacher_model_id,
    )
    write_json(args.report_out, out)

    print("\n=== Layer E Teacher Training Complete ===")
    print(f"Teacher model id: {out['model']['model_id']}")
    print(f"Artifacts: {out['model']['artifact_dir']}")
    print(f"Train rows: {out['model_info']['train_rows']}")
    print(f"Val rows: {out['model_info']['val_rows']}")
    print(f"Train loss: {out['metrics']['train']['train_loss']:.6f}")


if __name__ == "__main__":
    main()
