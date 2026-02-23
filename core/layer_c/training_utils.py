"""Utilities for training Layer C.

Pipeline: TF-IDF (word + char) → PCA (TruncatedSVD) → XGBoost.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from tqdm import tqdm
from xgboost import XGBClassifier

from core.settings import Settings as _Settings

log = logging.getLogger(__name__)

_settings = _Settings()
SEED = _settings.layer_c_seed

# cache pre-processed Layer-A/B filtered data
CACHE_DIR = Path(__file__).resolve().parent / "outputs" / ".cache"


def would_reach_layer_c(layer_a_result, layer_b_result):
    # Layer B hard-blocks never reach Layer C.
    if getattr(layer_b_result, "verdict", None) == "block":
        return False

    # SAFE allowlisting allows early exit only when Layer A is not suspicious.
    if (not getattr(layer_a_result, "suspicious", False)) and getattr(layer_b_result, "allowlisted", False):
        return False

    return True


def _cache_key(csv_path: str) -> str:
    """Produce a deterministic cache key from the CSV path + file content hash."""
    p = Path(csv_path)
    h = hashlib.md5()
    h.update(str(p.resolve()).encode())
    # Hash on file size + first/last 8 KB to avoid reading the whole file
    stat = p.stat()
    h.update(str(stat.st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(8192))
        if stat.st_size > 8192:
            f.seek(-8192, 2)
            h.update(f.read(8192))
    return h.hexdigest()


def load_data(csv_path, *, use_cache: bool = True):
    """Load dataset, run Layer A + B filtering, return (X, y, df).

    Results are cached to disk so subsequent runs skip the expensive
    per-row Layer-A/B inference. The cache is invalidated automatically
    when the source CSV changes (size or content hash).
    """
    from core.layer_a.pipeline import analyze_text
    from core.layer_b.signature_engine import SignatureEngine

    # --- Cache lookup -------------------------------------------------
    cache_path: Path | None = None
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _cache_key(csv_path)
        cache_path = CACHE_DIR / f"filtered_{key}.parquet"
        if cache_path.exists():
            log.info("Loading cached filtered data from %s", cache_path)
            print(f"[cache] Loading pre-filtered data from {cache_path.name}")
            used_df = pd.read_parquet(cache_path)
            X = used_df["processed_text"]
            y = used_df["label"]
            return X, y, used_df

    # --- Full pass through Layer A + B --------------------------------
    df = pd.read_csv(csv_path)
    y_all = df["label"].astype(int)

    layer_c_results = []
    signature_engine = SignatureEngine()

    allowlisted_allow = 0
    non_allowlisted_allow = 0

    for i in tqdm(range(len(df)), desc="Layer A+B filtering", unit="row"):
        layer_a_result = analyze_text(df["text"].iloc[i])
        layer_b_result = signature_engine.detect(layer_a_result.processed_text)

        if would_reach_layer_c(layer_a_result, layer_b_result):
            layer_c_results.append((layer_a_result.processed_text, y_all[i]))

            if layer_b_result.verdict == "allow" and not getattr(layer_b_result, "allowlisted", False):
                non_allowlisted_allow += 1
            if layer_b_result.verdict == "allow" and getattr(layer_b_result, "allowlisted", False):
                allowlisted_allow += 1

    X = pd.Series([t for (t, _) in layer_c_results], name="processed_text")
    y = pd.Series([lab for (_, lab) in layer_c_results], name="label")
    used_df = pd.DataFrame({"processed_text": X, "label": y})

    log.info(
        "Layer A+B filtering: %d → %d rows (allowlisted_allow=%d, non_allowlisted_allow=%d)",
        len(df), len(used_df), allowlisted_allow, non_allowlisted_allow,
    )
    print(
        f"Filtering complete: {len(df)} → {len(used_df)} rows "
        f"(allowlisted_allow={allowlisted_allow}, non_allowlisted_allow={non_allowlisted_allow})"
    )

    # --- Persist cache ------------------------------------------------
    if use_cache and cache_path is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        used_df.to_parquet(cache_path, index=False)
        print(f"[cache] Saved filtered data to {cache_path.name}")

    return X, y, used_df

def make_vectorizer():
    s = _settings
    word = TfidfVectorizer(
        ngram_range=s.layer_c_tfidf_word_ngram_range,
        analyzer="word",
        min_df=s.layer_c_tfidf_min_df,
        max_df=s.layer_c_tfidf_max_df,
        max_features=s.layer_c_tfidf_word_max_features,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        ngram_range=s.layer_c_tfidf_char_ngram_range,
        analyzer="char_wb",
        min_df=s.layer_c_tfidf_min_df,
        max_df=s.layer_c_tfidf_max_df,
        max_features=s.layer_c_tfidf_char_max_features,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])  # type: ignore

def make_reducer(n_components: int | None = None):
    """TruncatedSVD — the sparse-matrix equivalent of PCA (a.k.a. LSA)."""
    if n_components is None:
        n_components = _settings.layer_c_svd_components
    return TruncatedSVD(n_components=n_components, random_state=SEED)

def make_model(n_pos: int, n_neg: int):
    """Create XGBoost classifier with class imbalance handling."""
    s = _settings
    scale_pos = max(1.0, n_neg / max(1, n_pos))
    return XGBClassifier(
        n_estimators=s.layer_c_xgb_n_estimators,
        max_depth=s.layer_c_xgb_max_depth,
        learning_rate=s.layer_c_xgb_learning_rate,
        subsample=s.layer_c_xgb_subsample,
        colsample_bytree=s.layer_c_xgb_colsample_bytree,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        early_stopping_rounds=s.layer_c_xgb_early_stopping_rounds,
        n_jobs=-1,
        random_state=SEED,
    )


def save(vec, reducer, model, vectorizer_path, reducer_path, model_path):
    for p in (vectorizer_path, reducer_path, model_path):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(vec, vectorizer_path)
    # reducer may be None when SVD is skipped
    if reducer is not None:
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
    target_block_precision: float | None = None,
    max_malicious_allow_rate: float | None = None,
    min_flag_band: float | None = None,
):
    cfg = _settings
    if target_block_precision is None:
        target_block_precision = cfg.layer_c_tune_target_block_precision
    if max_malicious_allow_rate is None:
        max_malicious_allow_rate = cfg.layer_c_tune_max_malicious_allow_rate
    if min_flag_band is None:
        min_flag_band = cfg.layer_c_tune_min_flag_band

    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores)

    steps = cfg.layer_c_tune_grid_steps
    low_grid = np.linspace(cfg.layer_c_tune_low_grid_start, cfg.layer_c_tune_low_grid_end, steps)
    high_grid = np.linspace(cfg.layer_c_tune_high_grid_start, cfg.layer_c_tune_high_grid_end, steps)

    # Pre-compute label masks for vectorised scoring
    is_mal = y == 1
    mal_total = max(1, int(np.sum(is_mal)))

    best = None
    for low in low_grid:
        for high in high_grid:
            if low >= high or (high - low) < min_flag_band:
                continue

            # Three-way partition of traffic
            pred_allow = s < low
            pred_block = s >= high
            pred_flag = (~pred_allow) & (~pred_block)

            # Check block precision
            tp_block = np.sum(pred_block & is_mal)
            fp_block = np.sum(pred_block & (~is_mal))
            block_precision = (tp_block / (tp_block + fp_block)) if (tp_block + fp_block) else 1.0

            #if blocking isnt accurate enough, next pair pls
            if block_precision < target_block_precision:
                continue

            # Don't allow too many malicious through
            mal_allow_rate = np.sum(pred_allow & is_mal) / mal_total
            if mal_allow_rate > max_malicious_allow_rate:
                continue

            flag_rate = float(np.mean(pred_flag))
            block_rate = float(np.mean(pred_block))
            allow_rate = float(np.mean(pred_allow))
            block_recall = float(tp_block / mal_total)

            # Composite score: maximise block_recall + allow_rate while minimising flag_rate
            score = (
                cfg.layer_c_tune_w_block_recall * block_recall
                + cfg.layer_c_tune_w_allow_rate * allow_rate
                - cfg.layer_c_tune_w_flag_rate * flag_rate
            )

            if best is None or score > best[0]:
                best = (
                    score,
                    {
                        "low": float(low),
                        "high": float(high),
                        "val_flag_rate": flag_rate,
                        "val_block_rate": block_rate,
                        "val_allow_rate": allow_rate,
                        "val_block_precision": float(block_precision),
                        "val_block_recall": block_recall,
                        "val_malicious_allow_rate": float(mal_allow_rate),
                    },
                )
    #to return a default if no thresholds found
    if best is None:
        return {
            "low": _settings.layer_c_low_threshold,
            "high": _settings.layer_c_high_threshold,
            "val_flag_rate": 0.0,
            "val_block_rate": 0.0,
            "val_allow_rate": 0.0,
            "val_block_precision": 0.0,
            "val_block_recall": 0.0,
            "val_malicious_allow_rate": 0.0,
        }

    return best[1]


def train_eval(X, y, low=None, high=None):
    s = _settings
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=s.layer_c_val_test_size, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=s.layer_c_test_split, stratify=y_temp, random_state=SEED
    )

    # --- TF-IDF vectorisation -----------------------------------------
    print("Fitting TF-IDF vectorizer …")
    vec = make_vectorizer()
    X_train_tfidf = vec.fit_transform(X_train)
    X_val_tfidf = vec.transform(X_val)
    X_test_tfidf = vec.transform(X_test)
    print(f"TF-IDF features: {X_train_tfidf.shape[1]:,}")

    # --- Dimensionality reduction ------------------------------------
    n_components = min(_settings.layer_c_svd_components, X_train_tfidf.shape[1] - 1, X_train_tfidf.shape[0] - 1)
    print(f"Fitting TruncatedSVD ({n_components} components) …")
    reducer = make_reducer(n_components=n_components)
    X_train_vec = reducer.fit_transform(X_train_tfidf)
    X_val_vec = reducer.transform(X_val_tfidf)
    X_test_vec = reducer.transform(X_test_tfidf)

    explained_var = float(np.sum(reducer.explained_variance_ratio_))
    print(f"SVD: {X_train_tfidf.shape[1]:,} features → {n_components} components "
          f"({explained_var:.2%} variance explained)")

    # --- XGBoost ---
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    model = make_model(n_pos, n_neg)
    print("Training XGBoost …")
    model.fit(
        X_train_vec, y_train,
        eval_set=[(X_val_vec, y_val)],
        verbose=50,
    )
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        print(f"Early stopping: best iteration = {best_iteration}")

    val_scores = model.predict_proba(X_val_vec)[:, 1]
    test_scores = model.predict_proba(X_test_vec)[:, 1]
    val_pred_05 = val_scores >= 0.5
    test_pred_05 = test_scores >= 0.5

    tuned = None
    if low is None or high is None:
        print("Tuning routing thresholds …")
        tuned = tune_routing_thresholds(y_val.to_numpy(), val_scores)
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
