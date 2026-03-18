from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)

def save(artifact, model_path):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))


def route_to_label(scores, low, high):
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

def calibration_metrics(y_true, y_prob, bins: int):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    non_empty_bins = 0

    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue

        non_empty_bins += 1
        bin_acc = float(np.mean(y[mask]))
        bin_conf = float(np.mean(p[mask]))
        gap = abs(bin_acc - bin_conf)
        weight = float(np.sum(mask) / max(1, n))
        ece += gap * weight
        mce = max(mce, gap)

    return {
        "bins": int(bins),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece),
        "mce": float(mce),
        "non_empty_bins": int(non_empty_bins),
    }


def threshold_margin(scores, low: float, high: float):
    s = np.asarray(scores, dtype=float)
    low_dist = np.abs(s - low)
    high_dist = np.abs(s - high)
    nearest = np.minimum(low_dist, high_dist)
    return {
        "low_distance_mean": float(np.mean(low_dist)),
        "high_distance_mean": float(np.mean(high_dist)),
        "nearest_boundary_mean": float(np.mean(nearest)),
        "nearest_boundary_p10": float(np.percentile(nearest, 10)),
        "nearest_boundary_p50": float(np.percentile(nearest, 50)),
        "nearest_boundary_p90": float(np.percentile(nearest, 90)),
    }


def embedding_stats(emb):
    arr = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1)
    return {
        "rows": int(arr.shape[0]),
        "dim": int(arr.shape[1]),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
    }


def top_feature_importance(model, top_n: int = 20):
    importances = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
    if importances.size == 0:
        return []
    top_idx = np.argsort(importances)[::-1][:top_n]
    return [
        {
            "feature_index": int(i),
            "importance": float(importances[i]),
        }
        for i in top_idx
    ]


def pick_hard_negative_indices(
    y_train,
    train_scores,
    low: float,
    high: float,
    use_routing_band: bool,
    score_min: float,
    score_max: float,
    max_samples: int,
):
    y = np.asarray(y_train).astype(int)
    scores = np.asarray(train_scores, dtype=float)

    mine_min, mine_max = (float(low), float(high)) if use_routing_band else (float(score_min), float(score_max))
    if mine_min > mine_max:
        mine_min, mine_max = mine_max, mine_min

    safe_mask = y == 0
    band_mask = (scores >= mine_min) & (scores < mine_max)
    candidate_idx = np.where(safe_mask & band_mask)[0]

    if candidate_idx.size == 0:
        return np.array([], dtype=int), mine_min, mine_max

    ranked = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
    capped_max = int(max(0, max_samples))
    if capped_max > 0:
        ranked = ranked[:capped_max]

    return ranked.astype(int), mine_min, mine_max


def augment_with_hard_negatives(X_train, y_train, hard_idx, multiplier: int):
    if hard_idx.size == 0 or multiplier <= 0:
        return X_train, y_train

    extra_idx = np.tile(hard_idx, int(multiplier))
    X_aug = pd.concat([X_train, X_train.iloc[extra_idx]], ignore_index=True)
    y_aug = pd.concat([y_train, y_train.iloc[extra_idx]], ignore_index=True)
    return X_aug, y_aug


def build_split_metrics(y_true, raw_scores, calibrated_scores, low: float, high: float, cal_bins: int):
    y = np.asarray(y_true).astype(int)
    raw = np.asarray(raw_scores, dtype=float)
    cal = np.asarray(calibrated_scores, dtype=float)

    raw_pred = (raw >= 0.5).astype(int)
    cal_pred = (cal >= 0.5).astype(int)
    verdict, routed = route_to_label(cal, low=low, high=high)

    return {
        "raw": {
            "report_0.5": binary_report(y, raw_pred),
            "roc_auc": roc_auc_score(y, raw),
            "pr_auc": average_precision_score(y, raw),
        },
        "calibrated": {
            "report_0.5": binary_report(y, cal_pred),
            "roc_auc": roc_auc_score(y, cal),
            "pr_auc": average_precision_score(y, cal),
        },
        "calibration": calibration_metrics(y, cal, bins=cal_bins),
        "report_routing": binary_report(y, routed),
        "routing_verdict_counts": pd.Series(verdict).value_counts().to_dict(),
        "routing_verdict_by_label": verdict_breakdown(y, verdict),
        "routing_f1": float(f1_score(y, routed, zero_division=0)),
        "threshold_margin": threshold_margin(cal, low=low, high=high),
    }