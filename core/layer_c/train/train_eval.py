from core.settings import Settings
from core.layer_c.train.make_model import make_model
from core.layer_c.train.threshold_tuner import tune_routing_thresholds

from pathlib import Path
from sklearn.metrics import roc_auc_score,classification_report, f1_score
from sklearn.model_selection import train_test_split
import hashlib
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

_settings = Settings()
SEED = _settings.layer_c_seed

_EMB_CACHE_DIR = Path(__file__).resolve().parent / "outputs" / ".cache" / "embeddings"

def _emb_cache_path(texts, model_name: str) -> Path:
    """Return a deterministic cache path for a given text list + model."""
    h = hashlib.md5(model_name.encode())
    for t in texts:
        h.update(t.encode())
    return _EMB_CACHE_DIR / f"{h.hexdigest()}.npy"

def encode_texts(texts, model: SentenceTransformer, batch_size=None, use_cache = True):
    if batch_size is None:
        batch_size = _settings.layer_c_embedding_batch_size
    texts_list = list(texts)

    if use_cache:
        _EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _emb_cache_path(texts_list, _settings.layer_c_embedding_model)
        if cache_path.exists():
            print(f"[emb cache] Loading cached embeddings from {cache_path.name}")
            return np.load(cache_path)

    emb = model.encode(texts_list, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

    if use_cache:
        np.save(cache_path, emb) #type: ignore
        print(f"[emb cache] Saved embeddings to {cache_path.name}") #type: ignore

    return emb

def route_to_label(scores, low, high) :
    """Convert probabilities into a verdict."""

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

#main loop for training and evaluation
def train_eval(X, y, low=None, high=None):
    s = _settings
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=s.layer_c_val_test_size, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=s.layer_c_test_split, stratify=y_temp, random_state=SEED
    )

    # --- Sentence-transformer embeddings ------------------------------
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer '{s.layer_c_embedding_model}' on {_device} …")
    encoder = SentenceTransformer(s.layer_c_embedding_model, device=_device)
    emb_dim = encoder.get_sentence_embedding_dimension()

    print("Encoding training texts …")
    X_train_emb = encode_texts(X_train, encoder)
    print("Encoding validation texts …")
    X_val_emb = encode_texts(X_val, encoder)
    print("Encoding test texts …")
    X_test_emb = encode_texts(X_test, encoder)
    print(f"Embedding features: {emb_dim}")

    # --- XGBoost ---
    model = make_model()
    print("Training XGBoost …")
    model.fit(
        X_train_emb, y_train,
        eval_set=[(X_val_emb, y_val)],
        verbose=50,
    )
    if model.best_iteration is not None:
        print(f"Early stopping: best iteration = {model.best_iteration}")

    # move model to CPU for prediction & serialisation
    model.set_params(device="cpu")

    val_scores = model.predict_proba(X_val_emb)[:, 1]
    test_scores = model.predict_proba(X_test_emb)[:, 1]
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
        "model": model,
        "embedding_model": s.layer_c_embedding_model,
        "thresholds": {
            "low": float(low),
            "high": float(high),
            "tuning": (None if tuned is None else tuned),
        },
        "embedding_info": {"model": s.layer_c_embedding_model, "dim": emb_dim},
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