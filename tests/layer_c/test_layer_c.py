import sys
import json
from pathlib import Path
import io
from contextlib import redirect_stdout

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.layer_c.classifier import Classifier
from core.layer_a.pipeline import analyze_text
from core.layer_b.signature_engine import SignatureEngine
import pandas as pd
import numpy as np


REPORT_PATH = project_root / "test_results" / "layer_c_eval_latest.json"

ARTIFACTS = {
    "model_path": "core/layer_c/outputs/classifier.joblib",
}


def load_trained_thresholds():
    """Load the routing thresholds that were tuned during training."""
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text())
        t = report.get("thresholds", {})
        low = t.get("low", 0.25)
        high = t.get("high", 0.75)
        print(f"Loaded trained thresholds: low={low:.4f}, high={high:.4f}")
        return low, high
    print("WARNING: No training report found, using default thresholds")
    return 0.25, 0.75


def load_test_data(csv_path):
    df = pd.read_csv(csv_path)
    return df["text"].tolist(), df["label"].tolist()


def filter_through_layer_b(texts, labels):    
    layer_b = SignatureEngine()
    
    flagged_texts = []
    flagged_labels = []
        
    for idx, (text, label) in enumerate(zip(texts, labels)):
        if idx % 500 == 0:
            print(f"Processing {idx}/{len(texts)}...")
        
        # Layer A: Preprocess text
        with redirect_stdout(io.StringIO()):
            layer_a_result = analyze_text(text)
        
        # Layer B: Signature detection
        layer_b_result = layer_b.detect(layer_a_result.processed_text)
                
        # Only pass "flag" verdicts to classifier
        if layer_b_result.verdict == "flag":
            flagged_texts.append(layer_a_result.processed_text)
            flagged_labels.append(label)
    
    return flagged_texts, flagged_labels


def evaluate_classifier(classifier, texts, labels):
    results = []
    for idx, text in enumerate(texts):
        result = classifier.predict(text)
        results.append({
            'true_label': labels[idx],
            'verdict': result.verdict,
            'probability': result.probability_score,
        })

    df = pd.DataFrame(results)
    total = len(df)

    # ---- Confusion matrix (3-way routing) ----------------------------
    safe_allow = int(((df['true_label'] == 0) & (df['verdict'] == 'allow')).sum())
    safe_flag  = int(((df['true_label'] == 0) & (df['verdict'] == 'flag')).sum())
    safe_block = int(((df['true_label'] == 0) & (df['verdict'] == 'block')).sum())
    mal_allow  = int(((df['true_label'] == 1) & (df['verdict'] == 'allow')).sum())
    mal_flag   = int(((df['true_label'] == 1) & (df['verdict'] == 'flag')).sum())
    mal_block  = int(((df['true_label'] == 1) & (df['verdict'] == 'block')).sum())
    n_safe = safe_allow + safe_flag + safe_block
    n_mal  = mal_allow + mal_flag + mal_block

    print("\n" + "=" * 68)
    print("CONFUSION MATRIX")
    print("=" * 68)
    print(f"{'Ground Truth':<15} | {'Allow':>10} | {'Flag':>10} | {'Block':>10} | {'Total':>10}")
    print("-" * 68)
    print(f"{'SAFE':<15} | {safe_allow:>10} | {safe_flag:>10} | {safe_block:>10} | {n_safe:>10}")
    print(f"{'MALICIOUS':<15} | {mal_allow:>10} | {mal_flag:>10} | {mal_block:>10} | {n_mal:>10}")
    print("-" * 68)
    tot_allow = safe_allow + mal_allow
    tot_flag  = safe_flag + mal_flag
    tot_block = safe_block + mal_block
    print(f"{'Total':<15} | {tot_allow:>10} | {tot_flag:>10} | {tot_block:>10} | {total:>10}")
    print("=" * 68)

    # ---- Security metrics (what matters for a defence pipeline) ------
    # Binary view: flag + block = "detected", allow = "passed"
    tp = mal_flag + mal_block          # malicious correctly caught
    fp = safe_flag + safe_block        # safe incorrectly caught
    fn = mal_allow                     # malicious that slipped through
    tn = safe_allow                    # safe correctly passed

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / total if total else 0.0

    malicious_escape_rate = fn / n_mal if n_mal else 0.0
    safe_block_rate       = safe_block / n_safe if n_safe else 0.0
    safe_flag_rate        = safe_flag / n_safe if n_safe else 0.0
    flag_rate             = tot_flag / total if total else 0.0

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  (of all detected, fraction truly malicious)")
    print(f"Recall:    {recall:.4f}  (of all malicious, fraction detected)")
    print(f"F1:        {f1:.4f}")

    print("\nSecurity rates")
    print(f"Malicious escape rate (allowed malicious / all malicious): {malicious_escape_rate:.4f}  ({fn}/{n_mal})")
    print(f"Safe block rate       (blocked safe / all safe):           {safe_block_rate:.4f}  ({safe_block}/{n_safe})")
    print(f"Safe flag rate        (flagged safe / all safe):           {safe_flag_rate:.4f}  ({safe_flag}/{n_safe})")
    print(f"Overall flag rate     (flagged / total):                   {flag_rate:.4f}  ({tot_flag}/{total})")


def test_layer_c():
    test_texts, true_labels = load_test_data("datasets/barrikada_test.csv")
    
    # Filter through Layer A and B
    flagged_texts, flagged_labels = filter_through_layer_b(test_texts, true_labels)
    
    # Load thresholds from training report (not hardcoded)
    low, high = load_trained_thresholds()
    
    classifier = Classifier(
        **ARTIFACTS,
        low=low,
        high=high,
    )
    
    # Evaluate with trained thresholds
    evaluate_classifier(classifier, flagged_texts, flagged_labels)


if __name__ == "__main__":
    import time
    start_time = time.time()
    test_layer_c()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}s")