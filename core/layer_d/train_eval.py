import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)

from core.settings import Settings

_settings = Settings()
SEED = _settings.layer_c_seed


class LayerDTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, focal_gamma: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)

        losses = torch.nn.functional.cross_entropy(logits, labels, weight=weight, reduction="none")
        if self.focal_gamma > 0:
            pt = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            losses = losses * torch.pow(1.0 - pt.clamp(1e-6, 1.0), self.focal_gamma)

        loss = losses.mean()
        return (loss, outputs) if return_outputs else loss


def route_to_label(scores: np.ndarray, low: float, high: float):
    verdict = np.full(scores.shape, "allow")
    verdict[(scores >= low) & (scores < high)] = "flag"
    verdict[scores >= high] = "block"
    predicted_label = (verdict != "allow").astype(int)
    return verdict, predicted_label


def binary_report(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=False)


def verdict_breakdown(y_true, verdict):
    y = np.asarray(y_true).astype(int)
    v = np.asarray(verdict)
    out = {
        "allow": {"0": 0, "1": 0},
        "flag": {"0": 0, "1": 0},
        "block": {"0": 0, "1": 0},
    }
    for label in (0, 1):
        for decision in ("allow", "flag", "block"):
            out[decision][str(label)] = int(np.sum((y == label) & (v == decision)))
    return out


def make_compute_metrics(low: float, high: float):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        scores = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        labels_arr = np.asarray(labels).astype(int)
        preds = (scores >= 0.5).astype(int)

        tp = int(np.sum((preds == 1) & (labels_arr == 1)))
        fp = int(np.sum((preds == 1) & (labels_arr == 0)))
        fn = int(np.sum((preds == 0) & (labels_arr == 1)))

        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        acc = float(np.mean(preds == labels_arr))

        verdict, routed = route_to_label(scores, low=low, high=high)
        routed_tp = int(np.sum((routed == 1) & (labels_arr == 1)))
        routed_fp = int(np.sum((routed == 1) & (labels_arr == 0)))
        routed_fn = int(np.sum((routed == 0) & (labels_arr == 1)))
        routed_precision = routed_tp / max(1, routed_tp + routed_fp)
        routed_recall = routed_tp / max(1, routed_tp + routed_fn)
        routed_f1 = 2 * routed_precision * routed_recall / max(1e-12, routed_precision + routed_recall)
        safe_fpr = float(np.mean((verdict != "allow") & (labels_arr == 0)))
        mal_allow = float(np.mean((verdict == "allow") & (labels_arr == 1)) / max(1e-12, np.mean(labels_arr == 1)))

        metrics = {
            "accuracy": acc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": float(average_precision_score(labels_arr, scores)),
            "routing_precision": float(routed_precision),
            "routing_recall": float(routed_recall),
            "routing_f1": float(routed_f1),
            "routing_safe_fpr": safe_fpr,
            "routing_malicious_allow_rate": mal_allow,
            "routing_low": float(low),
            "routing_high": float(high),
            "security_score": float(routed_f1 - (0.5 * mal_allow) - (0.25 * safe_fpr)),
        }

        try:
            metrics["roc_auc"] = float(roc_auc_score(labels_arr, scores))
        except ValueError:
            pass
        return metrics

    return compute_metrics


def tokenize_datasets(tokenizer, train_df, val_df, test_df, max_length: int):
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    train_ds = train_ds.map(tokenize, batched=True, num_proc=4)
    val_ds = val_ds.map(tokenize, batched=True, num_proc=4)
    test_ds = test_ds.map(tokenize, batched=True, num_proc=4)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_ds, val_ds, test_ds


def predict_scores(trainer: Trainer, ds) -> np.ndarray:
    pred = trainer.predict(ds)
    logits = pred.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return probs


def train_eval(X, y, model_out_dir: str, low=None, high=None):
    s = _settings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    split_train = s.layer_d_split_train
    split_val = s.layer_d_split_val
    split_test = s.layer_d_split_test
    if abs((split_train + split_val + split_test) - 1.0) > 1e-6:
        raise ValueError("Layer D splits must sum to 1.0")

    val_test_ratio = split_val + split_test
    test_ratio_of_temp = split_test / max(1e-12, val_test_ratio)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_ratio, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio_of_temp, stratify=y_temp, random_state=SEED
    )

    train_df = pd.DataFrame({"text": X_train.tolist(), "label": y_train.astype(int).tolist()})
    val_df = pd.DataFrame({"text": X_val.tolist(), "label": y_val.astype(int).tolist()})
    test_df = pd.DataFrame({"text": X_test.tolist(), "label": y_test.astype(int).tolist()})

    print(f"Loading tokenizer '{s.layer_d_model_id}' ...")
    tokenizer = AutoTokenizer.from_pretrained(s.layer_d_model_id)

    print("Tokenizing train/val/test datasets ...")
    train_ds, val_ds, test_ds = tokenize_datasets(
        tokenizer,
        train_df,
        val_df,
        test_df,
        max_length=s.layer_d_max_length,
    )

    model_kwargs = {
        "num_labels": 2,
        "id2label": {0: "SAFE", 1: "INJECTION"},
        "label2id": {"SAFE": 0, "INJECTION": 1},
    }

    if torch.cuda.is_available() and s.layer_d_use_bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16

    try:
        print("Loading Layer D ModernBERT with flash_attention_2 ...")
        model = AutoModelForSequenceClassification.from_pretrained(
            s.layer_d_model_id,
            attn_implementation="flash_attention_2",
            **model_kwargs,
        )
    except Exception:
        print("flash_attention_2 unavailable; loading default attention ...")
        model = AutoModelForSequenceClassification.from_pretrained(
            s.layer_d_model_id,
            **model_kwargs,
        )

    if torch.cuda.is_available() and s.layer_d_use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(model_out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    low_threshold = float(s.layer_d_low_threshold if low is None else low)
    high_threshold = float(s.layer_d_high_threshold if high is None else high)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=s.layer_d_num_train_epochs,
        learning_rate=s.layer_d_learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=s.layer_d_warmup_ratio,
        per_device_train_batch_size=s.layer_d_train_batch_size,
        per_device_eval_batch_size=s.layer_d_eval_batch_size,
        gradient_accumulation_steps=s.layer_d_gradient_accumulation_steps,
        gradient_checkpointing=s.layer_d_gradient_checkpointing,
        weight_decay=s.layer_d_weight_decay,
        max_grad_norm=1.0,
        bf16=(torch.cuda.is_available() and s.layer_d_use_bf16),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_security_score",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=100,
        dataloader_num_workers=s.layer_d_dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=SEED,
        report_to=[],
    )

    class_weights = torch.tensor(
        [1.0, float(s.layer_d_malicious_class_weight)],
        dtype=torch.float32,
    )

    trainer = LayerDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        compute_metrics=make_compute_metrics(low_threshold, high_threshold),
        class_weights=class_weights,
        focal_gamma=s.layer_d_focal_gamma,
    )

    print("Training Layer D ModernBERT ...")
    trainer.train()

    print(f"Saving model artifacts to {output_dir} ...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Scoring validation and test sets ...")
    val_scores = predict_scores(trainer, val_ds)
    test_scores = predict_scores(trainer, test_ds)

    val_pred_05 = val_scores >= 0.5
    test_pred_05 = test_scores >= 0.5

    val_verdict, val_pred_route = route_to_label(val_scores, low=low_threshold, high=high_threshold)
    test_verdict, test_pred_route = route_to_label(test_scores, low=low_threshold, high=high_threshold)

    val_verdict_counts = pd.Series(val_verdict).value_counts().to_dict()
    test_verdict_counts = pd.Series(test_verdict).value_counts().to_dict()

    return {
        "model": {
            "type": "modernbert_sequence_classifier",
            "model_id": s.layer_d_model_id,
            "artifact_dir": str(output_dir),
            "max_length": s.layer_d_max_length,
        },
        "thresholds": {
            "low": low_threshold,
            "high": high_threshold,
        },
        "model_info": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "model_id": s.layer_d_model_id,
            "max_length": s.layer_d_max_length,
        },
        "metrics": {
            "val": {
                "roc_auc": float(roc_auc_score(y_val, val_scores)),
                "report_0.5": binary_report(y_val, val_pred_05),
                "report_routing": binary_report(y_val, val_pred_route),
                "routing_verdict_counts": val_verdict_counts,
                "routing_verdict_by_label": verdict_breakdown(y_val.to_numpy(), val_verdict),
                "routing_f1": float(f1_score(y_val.to_numpy(), val_pred_route, zero_division=0)),
            },
            "test": {
                "roc_auc": float(roc_auc_score(y_test, test_scores)),
                "report_0.5": binary_report(y_test, test_pred_05),
                "report_routing": binary_report(y_test, test_pred_route),
                "routing_verdict_counts": test_verdict_counts,
                "routing_verdict_by_label": verdict_breakdown(y_test.to_numpy(), test_verdict),
                "routing_f1": float(f1_score(y_test.to_numpy(), test_pred_route, zero_division=0)),
            },
        },
    }
