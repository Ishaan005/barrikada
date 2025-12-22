"""Small training script for Layer C (Tier-2 ML).

Baseline: TF-IDF (word + char) + Logistic Regression.
Saves artifacts + prints metrics + writes top features.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

# Allow running as a script: `python core/layer_c/train.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.settings import Settings

SEED = 42

def load_data(csv_path: str):
    from core.layer_a.pipeline import analyze_text
    from core.layer_b.signature_engine import SignatureEngine

    df = pd.read_csv(csv_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Columns: text,label not found! :(")

    X = df["text"]
    y = df["label"]

    # Preprocess + filter to only unsure cases that reach Layer C
    layer_b_results = []
    signature_engine = SignatureEngine()
    allow = flag = block = 0
    for i in range(len(df)):
        layer_a_result = analyze_text(df["text"].iloc[i])

        layer_b_result = signature_engine.detect(layer_a_result.processed_text)
        if layer_b_result.verdict in ["allow"]:
            allow += 1
        elif layer_b_result.verdict in ["block"]:
            block += 1
        elif layer_b_result.verdict in ["flag"]:
            flag += 1

        if layer_b_result.verdict in ["flag"]:
            layer_b_results.append([layer_a_result.processed_text, y[i]])
        
    print(f"Filtered to {len(layer_b_results)} samples that reach Layer C.")
    #print(f"Layer B breakdown: allow={allow}, block={block}, flag={flag}")
    
    X = pd.Series([item[0] for item in layer_b_results])
    y = pd.Series([item[1] for item in layer_b_results])
    layer_b_df = pd.DataFrame(layer_b_results, columns=["text", "label"])

    return X, y, layer_b_df


def make_vectorizer():
    word = TfidfVectorizer(
        ngram_range=(1, 2), 
        analyzer="word", 
        min_df=2)
    
    char = TfidfVectorizer(
        ngram_range=(3, 5), 
        analyzer="char_wb", 
        min_df=2)
    
    return FeatureUnion([("word", word), ("char", char)])#type: ignore


def make_model():
    return LogisticRegression(
        solver="saga",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )

def save(vec: FeatureUnion, model: LogisticRegression, vectorizer_path: str, model_path: str):
    Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, vectorizer_path)
    joblib.dump(model, model_path)


def top_features(vec: FeatureUnion, model: LogisticRegression, k: int = 30):
    names = []
    for prefix, transformer in vec.transformer_list:  # type: ignore[attr-defined]
        feats = transformer.get_feature_names_out()
        names.extend([f"{prefix}__{f}" for f in feats])

    coef = model.coef_.ravel()
    pos = np.argsort(coef)[::-1][:k]
    neg = np.argsort(coef)[:k]
    return {
        "top_positive": [{"feature": names[i], "weight": float(coef[i])} for i in pos],
        "top_negative": [{"feature": names[i], "weight": float(coef[i])} for i in neg],
    }


def train_eval(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    vec = make_vectorizer()
    X_train_vec = vec.fit_transform(X_train)
    X_val_vec = vec.transform(X_val)
    X_test_vec = vec.transform(X_test)

    model = make_model()
    model.fit(X_train_vec, y_train)

    val_scores = model.predict_proba(X_val_vec)[:, 1]
    test_scores = model.predict_proba(X_test_vec)[:, 1]
    val_pred = (val_scores >= 0.5).astype(int)
    test_pred = (test_scores >= 0.5).astype(int)

    return {
        "vectorizer": vec,
        "model": model,
        "metrics": {
            "val": {
                "roc_auc": float(roc_auc_score(y_val, val_scores)),
                "report": classification_report(y_val, val_pred, digits=4, zero_division=0),
            },
            "test": {
                "roc_auc": float(roc_auc_score(y_test, test_scores)),
                "report": classification_report(y_test, test_pred, digits=4, zero_division=0),
            },
        },
    }


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))


def main() -> int:
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
    parser.add_argument(
        "--explain-out",
        default=str(Path("test_results") / "layer_c_top_features_latest.json"),
        help="Path to write top-feature explainability report (.json)",
    )

    args = parser.parse_args()

    # Default behavior: train only on samples that would reach Layer C.
    X, y, df = load_data(args.csv)
    # out = train_eval(X, y)

    # vec = out["vectorizer"]
    # model = out["model"]
    # metrics = out["metrics"]

    # save(vec, model, args.vectorizer_out, args.model_out)
    # _write_json(args.report_out, metrics)
    # _write_json(args.explain_out, {"top_features": top_features(vec, model, k=30)})

    # # Console output for quick feedback
    # print("\n=== Layer C (Tier-2) Evaluation ===")
    # print("VAL:\n" + metrics["val"]["report"])
    # print("TEST:\n" + metrics["test"]["report"])
    # print(f"VAL ROC-AUC: {metrics['val']['roc_auc']:.4f}")
    # print(f"TEST ROC-AUC: {metrics['test']['roc_auc']:.4f}")
    # print(f"Artifacts saved:\n- {args.vectorizer_out}\n- {args.model_out}")
    # print(f"Reports saved:\n- {args.report_out}\n- {args.explain_out}")
    # print(f"Rows used: {len(df)} (seed={SEED})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
