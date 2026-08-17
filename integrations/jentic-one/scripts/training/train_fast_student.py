"""Train a compact Layer D student with optional soft-label distillation.

This developer-only pipeline refuses test-split data and does not read the Jentic release
corpus. It creates a prototype candidate; promotion still requires the independent corpus,
ONNX parity, latency, and shadow gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from core.release_policy import (  # noqa: E402
    MAX_BENIGN_FALSE_BLOCK_RATE,
    MAX_BENIGN_FLAG_RATE,
    MIN_ATTACK_RECALL,
)

EXPECTED_FIELDS = {"id", "text", "label", "surface", "family", "split", "provenance"}


def load_student_rows(path: Path) -> list[dict]:
    rows = []
    seen_ids = set()
    seen_digests = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or set(raw) != EXPECTED_FIELDS:
                raise ValueError(f"line {line_number}: student data has a closed schema")
            if raw["split"] not in {"train", "validation"}:
                raise ValueError(f"line {line_number}: test data is forbidden in student training")
            if raw["label"] not in {0, 1}:
                raise ValueError(f"line {line_number}: label must be 0 or 1")
            if raw["surface"] not in {"runtime_response", "specification"}:
                raise ValueError(f"line {line_number}: invalid surface")
            if not isinstance(raw["text"], str) or not raw["text"].strip():
                raise ValueError(f"line {line_number}: text is empty")
            digest = hashlib.sha256(" ".join(raw["text"].casefold().split()).encode()).hexdigest()
            if raw["id"] in seen_ids or digest in seen_digests:
                raise ValueError(f"line {line_number}: duplicate id or normalized text")
            seen_ids.add(raw["id"])
            seen_digests.add(digest)
            rows.append(raw)
    if not rows:
        raise ValueError("student training data is empty")
    combinations = Counter((row["split"], row["label"]) for row in rows)
    if any(
        combinations[(split, label)] == 0 for split in ("train", "validation") for label in (0, 1)
    ):
        raise ValueError("student data must contain both labels in train and validation splits")
    return rows


def _teacher_scores(rows: list[dict], teacher_dir: Path, batch_size: int) -> list[float]:
    from core.layer_d.classifier import LayerDClassifier  # noqa: PLC0415

    teacher = LayerDClassifier(teacher_dir, low=0.05, high=0.95, max_length=512)
    scores = []
    for start in range(0, len(rows), batch_size):
        texts = [row["text"] for row in rows[start : start + batch_size]]
        scores.extend(float(value) for value in teacher.predict_batch(texts))
    return scores


def _rates(labels: np.ndarray, scores: np.ndarray, low: float, high: float) -> dict:
    allow = scores < low
    block = scores >= high
    flag = (~allow) & (~block)
    attacks = labels == 1
    benign = labels == 0
    return {
        "attack_recall": float(np.mean(~allow[attacks])),
        "attack_block_rate": float(np.mean(block[attacks])),
        "benign_false_block_rate": float(np.mean(block[benign])),
        "benign_flag_rate": float(np.mean(flag[benign])),
    }


def calibrate_thresholds(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, dict]:
    # A fixed grid keeps calibration bounded even for large validation sets. The rates that
    # depend only on one threshold are precomputed; pair selection is then constant-time.
    candidates = np.linspace(0.001, 0.999, num=999)
    attacks = labels == 1
    benign = labels == 0
    low_rates = {
        float(threshold): (
            float(np.mean(scores[attacks] >= threshold)),
            float(np.mean(scores[benign] >= threshold)),
        )
        for threshold in candidates
    }
    high_rates = {
        float(threshold): (
            float(np.mean(scores[attacks] >= threshold)),
            float(np.mean(scores[benign] >= threshold)),
        )
        for threshold in candidates
    }
    valid_pairs = []
    all_pairs = []
    for low in candidates:
        for high in candidates:
            if low >= high:
                continue
            attack_recall, benign_nonallow = low_rates[float(low)]
            attack_block, benign_block = high_rates[float(high)]
            rates = {
                "attack_recall": attack_recall,
                "attack_block_rate": attack_block,
                "benign_false_block_rate": benign_block,
                "benign_flag_rate": max(0.0, benign_nonallow - benign_block),
            }
            quality = (
                rates["attack_recall"]
                + rates["attack_block_rate"]
                - (4 * rates["benign_false_block_rate"])
                - (2 * rates["benign_flag_rate"])
            )
            item = (quality, low, high, rates)
            all_pairs.append(item)
            if (
                rates["attack_recall"] >= MIN_ATTACK_RECALL
                and rates["benign_false_block_rate"] <= MAX_BENIGN_FALSE_BLOCK_RATE
                and rates["benign_flag_rate"] <= MAX_BENIGN_FLAG_RATE
            ):
                valid_pairs.append(item)
    selected = max(valid_pairs or all_pairs, key=lambda item: (item[0], item[1], -item[2]))
    _, low, high, rates = selected
    rates["validation_gates_passed"] = bool(valid_pairs)
    return low, high, rates


def train(args) -> dict:
    import torch  # noqa: PLC0415
    import torch.nn.functional as functional  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    rows = load_student_rows(args.data)
    if args.teacher_dir:
        scores = _teacher_scores(rows, args.teacher_dir, args.teacher_batch_size)
        for row, teacher_score in zip(rows, scores, strict=True):
            row["teacher_score"] = teacher_score

    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        args.student_model,
        local_files_only=not args.allow_download,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(  # nosec B615
        args.student_model,
        num_labels=2,
        id2label={0: "SAFE", 1: "INJECTION"},
        label2id={"SAFE": 0, "INJECTION": 1},
        local_files_only=not args.allow_download,
        trust_remote_code=False,
    )

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]

    def make_dataset(source_rows):
        records = [
            {
                "text": row["text"],
                "labels": row["label"],
                **({"teacher_score": row["teacher_score"]} if "teacher_score" in row else {}),
            }
            for row in source_rows
        ]
        dataset = Dataset.from_list(records)

        def tokenize(batch):
            return tokenizer(
                batch["text"], truncation=True, max_length=args.max_length, padding=False
            )

        return dataset.map(tokenize, batched=True, remove_columns=["text"])

    train_dataset = make_dataset(train_rows)
    val_dataset = make_dataset(val_rows)

    class StudentTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            teacher_score = inputs.pop("teacher_score", None)
            outputs = model(**inputs)
            supervised = functional.cross_entropy(outputs.logits, inputs["labels"])
            loss = supervised
            if teacher_score is not None and args.distillation_alpha > 0:
                teacher_score = (
                    teacher_score.to(outputs.logits.device).float().clamp(1e-5, 1 - 1e-5)
                )
                teacher_probs = torch.stack((1 - teacher_score, teacher_score), dim=-1)
                student_log_probs = functional.log_softmax(
                    outputs.logits / args.temperature, dim=-1
                )
                distilled = functional.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                ) * (args.temperature**2)
                loss = (
                    1 - args.distillation_alpha
                ) * supervised + args.distillation_alpha * distilled
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(args.output),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        logging_strategy="epoch",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    trainer = StudentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    prediction = trainer.predict(val_dataset)
    logits = np.asarray(prediction.predictions, dtype=float)
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    scores = probabilities[:, 1] / probabilities.sum(axis=-1)
    labels = np.asarray([row["label"] for row in val_rows], dtype=int)
    low, high, rates = calibrate_thresholds(labels, scores)
    source_digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    report = {
        "model_type": "fast_layer_d_student",
        "base_model": args.student_model,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "source_sha256": source_digest,
        "source_contains_test_split": False,
        "distillation": {
            "enabled": bool(args.teacher_dir),
            "alpha": args.distillation_alpha if args.teacher_dir else 0.0,
            "temperature": args.temperature if args.teacher_dir else None,
        },
        "validation_roc_auc": float(roc_auc_score(labels, scores)),
        "thresholds": {"low": low, "high": high},
        "validation_rates": rates,
    }
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-model", default="distilbert-base-uncased")
    parser.add_argument("--teacher-dir", type=Path)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--distillation-alpha", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.distillation_alpha < 1:
        parser.error("--distillation-alpha must be in [0, 1)")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    report = train(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
