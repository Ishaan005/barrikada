"""Utilities for training Layer C.

Pipeline: TF-IDF (word + char) → PCA (TruncatedSVD) → XGBoost.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from xgboost import XGBClassifier

SEED = 42


def would_reach_layer_c(layer_a_result, layer_b_result):
    # Layer B hard-blocks never reach Layer C.
    if getattr(layer_b_result, "verdict", None) == "block":
        return False

    # SAFE allowlisting allows early exit only when Layer A is not suspicious.
    if (not getattr(layer_a_result, "suspicious", False)) and getattr(layer_b_result, "allowlisted", False):
        return False

    return True


def load_data(csv_path):
    from core.layer_a.pipeline import analyze_text
    from core.layer_b.signature_engine import SignatureEngine

    df = pd.read_csv(csv_path)

    y = df["label"].astype(int)

    layer_c_results = []
    signature_engine = SignatureEngine()

    allowlisted_allow = 0
    non_allowlisted_allow = 0

    for i in range(len(df)):
        layer_a_result = analyze_text(df["text"].iloc[i])

        layer_b_result = signature_engine.detect(layer_a_result.processed_text)

        if would_reach_layer_c(layer_a_result, layer_b_result):
            layer_c_results.append((layer_a_result.processed_text, y[i]))

            if layer_b_result.verdict == "allow" and not getattr(layer_b_result, "allowlisted", False):
                non_allowlisted_allow += 1
            if layer_b_result.verdict == "allow" and getattr(layer_b_result, "allowlisted", False):
                allowlisted_allow += 1

    X = pd.Series([t for (t, _) in layer_c_results], name="processed_text")
    y = pd.Series([lab for (_, lab) in layer_c_results], name="label")
    used_df = pd.DataFrame({"processed_text": X, "label": y})

    return X, y, used_df

def make_vectorizer():
    word = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=2)
    char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=2)
    return FeatureUnion([("word", word), ("char", char)])  # type: ignore

def make_reducer(n_components: int = 200):
    """TruncatedSVD — the sparse-matrix equivalent of PCA (a.k.a. LSA)."""
    return TruncatedSVD(n_components=n_components, random_state=SEED)

def make_model(n_pos: int, n_neg: int):
    """Create XGBoost classifier with class imbalance handling."""
    scale_pos = max(1.0, n_neg / max(1, n_pos))
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=SEED,
    )


def save(vec, reducer, model, vectorizer_path, reducer_path, model_path):
    for p in (vectorizer_path, reducer_path, model_path):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(vec, vectorizer_path)
    joblib.dump(reducer, reducer_path)
    joblib.dump(model, model_path)


def binary_report(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=False)


def route_to_label(scores, low, high) :
    """Convert probabilities into a verdict.
    """

    verdict = np.full(scores.shape, "allow")
    verdict[(scores >= low) & (scores < high)] = "flag"
    verdict[scores >= high] = "block"

    predicted_label = (verdict != "allow").astype(int)

    return verdict, predicted_label


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


def tune_routing_thresholds(
    y_true,
    scores,
    target_block_precision = 0.99,
    max_malicious_allow_rate = 0.02,
    min_flag_band = 0.05,
):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores)

    low_grid = np.linspace(0.05, 0.60, 100)
    high_grid = np.linspace(0.40, 0.99, 100)

    best = None
    for low in low_grid:
        for high in high_grid:
            if low >= high or (high - low) < min_flag_band:
                continue
            
            #three-way partition of traffic
            pred_allow = s < low #true if score < low
            pred_block = s >= high
            pred_flag = (~pred_allow) & (~pred_block)

            #check block precision 
            tp_block = sum(pred_block & (y == 1))
            fp_block = sum(pred_block & (y == 0))
            block_precision = (tp_block / (tp_block + fp_block)) if (tp_block + fp_block) else 1.0

            #if blocking isnt accurate enough, next pair pls
            if block_precision < target_block_precision:
                continue

            # dont't allow too many malicious
            mal_total = max(1, np.sum(y == 1))
            mal_allow_rate = np.sum(pred_allow & (y == 1)) / mal_total
            if mal_allow_rate > max_malicious_allow_rate:
                continue

            flag_rate = np.mean(pred_flag)
            block_rate = np.mean(pred_block)
            allow_rate = np.mean(pred_allow)

            block_recall = tp_block / mal_total

            # ranking key: minimize flag_rate, then block_rate, then maximize allow_rate
            key = (flag_rate, block_rate, -allow_rate)
            if best is None or key < best[0]:
                best = (
                    key,
                    {
                        "low": float(low),
                        "high": float(high),
                        "val_flag_rate": float(flag_rate),
                        "val_block_rate": float(block_rate),
                        "val_allow_rate": float(allow_rate),
                        "val_block_precision": float(block_precision),
                        "val_block_recall": float(block_recall),
                        "val_malicious_allow_rate": float(mal_allow_rate),
                    },
                )
    #to return a default if no thresholds found
    if best is None:
        return {
            "low": 0.25,
            "high": 0.75,
            "val_flag_rate": 0.0,
            "val_block_rate": 0.0,
            "val_allow_rate": 0.0,
            "val_block_precision": 0.0,
            "val_block_recall": 0.0,
            "val_malicious_allow_rate": 0.0,
        }

    return best[1]


def train_eval(X, y, low = None, high = None):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    # tfidf vectorizer
    vec = make_vectorizer()
    X_train_tfidf = vec.fit_transform(X_train)
    X_val_tfidf = vec.transform(X_val)
    X_test_tfidf = vec.transform(X_test)

    # pca
    n_components = min(200, X_train_tfidf.shape[1] - 1, X_train_tfidf.shape[0] - 1)
    reducer = make_reducer(n_components=n_components)
    X_train_vec = reducer.fit_transform(X_train_tfidf)
    X_val_vec = reducer.transform(X_val_tfidf)
    X_test_vec = reducer.transform(X_test_tfidf)

    explained_var = float(np.sum(reducer.explained_variance_ratio_))
    print(f"PCA: {X_train_tfidf.shape[1]} features → {n_components} components "
          f"({explained_var:.2%} variance explained)")

    # --- XGBoost ---
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    model = make_model(n_pos, n_neg)
    model.fit(X_train_vec, y_train)

    val_scores = model.predict_proba(X_val_vec)[:, 1]
    test_scores = model.predict_proba(X_test_vec)[:, 1]
    val_pred_05 = val_scores >= 0.5
    test_pred_05 = test_scores >= 0.5

    tuned = None
    if low is None or high is None:
        tuned = tune_routing_thresholds(y_val.to_numpy(),val_scores)
        low = float(tuned["low"])
        high = float(tuned["high"])

    val_verdict, val_pred_route = route_to_label(val_scores, low=low, high=high)
    test_verdict, test_pred_route = route_to_label(test_scores, low=low, high=high)

    val_verdict_counts = pd.Series(val_verdict).value_counts().to_dict()
    test_verdict_counts = pd.Series(test_verdict).value_counts().to_dict()

    return {
        "vectorizer": vec,
        "reducer": reducer,
        "model": model,
        "thresholds": {
            "low": float(low),
            "high": float(high),
            "tuning": (None if tuned is None else tuned),
        },
        "pca": {
            "n_components": n_components,
            "explained_variance": explained_var,
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


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))
