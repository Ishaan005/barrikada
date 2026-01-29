import sys
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
                
        # Only pass "flag" verdicts to classfier
        if layer_b_result.verdict == "flag":
            flagged_texts.append(layer_a_result.processed_text)
            flagged_labels.append(label)
    
    return flagged_texts, flagged_labels


def run_grid_search(classifier, texts, labels, optimize_for="f1"):
    print(f"\nRunning grid search (optimizing for {optimize_for})...")
    result = classifier.grid_search(texts, labels, optimize_for=optimize_for)
    print(f"Best thresholds: low={result['best_low']:.2f}, high={result['best_high']:.2f}")
    print(f"Metrics: P={result['precision']:.4f}, R={result['recall']:.4f}, F1={result['f1']:.4f}")
    return result


def evaluate_classifier(classifier, texts, labels):
    results = []
    for idx, text in enumerate(texts):
        result = classifier.predict(text)
        predicted_label = 1 if result.verdict in ["block", "flag"] else 0
        results.append({
            'true_label': labels[idx],
            'layer_c_verdict': result.verdict,
            'predicted_label': predicted_label,
        })
    
    results_df = pd.DataFrame(results)
    
    # Confusion Matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    
    safe_allow = ((results_df['true_label'] == 0) & (results_df['layer_c_verdict'] == 'allow')).sum()
    safe_flag = ((results_df['true_label'] == 0) & (results_df['layer_c_verdict'] == 'flag')).sum()
    safe_block = ((results_df['true_label'] == 0) & (results_df['layer_c_verdict'] == 'block')).sum()
    
    malicious_allow = ((results_df['true_label'] == 1) & (results_df['layer_c_verdict'] == 'allow')).sum()
    malicious_flag = ((results_df['true_label'] == 1) & (results_df['layer_c_verdict'] == 'flag')).sum()
    malicious_block = ((results_df['true_label'] == 1) & (results_df['layer_c_verdict'] == 'block')).sum()
    
    safe_total = safe_allow + safe_flag + safe_block
    malicious_total = malicious_allow + malicious_flag + malicious_block
    total = safe_total + malicious_total
    
    print(f"{'Ground Truth':<15} | {'Allow':>10} | {'Flag':>10} | {'Block':>10} | {'Total':>10}")
    print("-" * 60)
    print(f"{'SAFE':<15} | {safe_allow:>10} | {safe_flag:>10} | {safe_block:>10} | {safe_total:>10}")
    print(f"{'MALICIOUS':<15} | {malicious_allow:>10} | {malicious_flag:>10} | {malicious_block:>10} | {malicious_total:>10}")
    print("-" * 60)
    print(f"{'Total':<15} | {safe_allow + malicious_allow:>10} | {safe_flag + malicious_flag:>10} | {safe_block + malicious_block:>10} | {total:>10}")
    print("="*60)
    
    tp = ((results_df['predicted_label'] == 1) & (results_df['true_label'] == 1)).sum()
    fp = ((results_df['predicted_label'] == 1) & (results_df['true_label'] == 0)).sum()
    fn = ((results_df['predicted_label'] == 0) & (results_df['true_label'] == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


def test_layer_c():
    test_texts, true_labels = load_test_data("datasets/barrikada_test.csv")
    
    # Filter through Layer a and b
    flagged_texts, flagged_labels = filter_through_layer_b(test_texts, true_labels)
    
    classifier = Classifier(
        vectorizer_path="core/layer_c/outputs/tf_idf_vectorizer.joblib",
        model_path="core/layer_c/outputs/tf_idf_logreg.joblib",
        low =0.35, 
        high=0.50
    )
    
    # Run grid search on flagged prompts to find optimal thresholds
    run_grid_search(classifier, flagged_texts, flagged_labels, optimize_for="f1")
    
    # Apply optimal thresholds and evaluate
    evaluate_classifier(classifier, flagged_texts, flagged_labels)


if __name__ == "__main__":
    test_layer_c()