from contextlib import redirect_stdout
import io
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.layer_c.classifier import Classifier
from core.layer_a.pipeline import analyze_text
from core.layer_b.signature_engine import SignatureEngine
import pandas as pd
import time

def load_test_data(csv_path):
    df = pd.read_csv(csv_path)
    return df["text"].tolist(), df["label"].tolist()

def filter_through_layer_b(texts, labels):    
    layer_b = SignatureEngine()
    
    flagged_texts = []
    flagged_labels = []
        
    layer_b_start = time.time()
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

    layer_b_end = time.time()
    layer_b_processing_time_s = (layer_b_end - layer_b_start)
    print(f"Layer B processing time (s): {layer_b_processing_time_s}")
    
    return flagged_texts, flagged_labels

def filter_through_layer_c(texts, labels):
    layer_c = Classifier(
        vectorizer_path="core/layer_c/outputs/tf_idf_vectorizer.joblib",
        model_path="core/layer_c/outputs/tf_idf_logreg.joblib",
        low =0.35, 
        high=0.50
    )
    
    flagged_texts = []
    flagged_labels = []
    layer_c_start = time.time()
    for idx, (text, label) in enumerate(zip(texts, labels)):
        if idx % 500 == 0:
            print(f"Processing {idx}/{len(texts)}...")
        
        # Layer C: Classifier prediction
        layer_c_result = layer_c.predict(text)
        # Only pass "flag" verdicts to Layer E
        if layer_c_result.verdict == "flag":
            flagged_texts.append(text)
            flagged_labels.append(label)
    layer_c_end = time.time()
    layer_c_processing_time_s = (layer_c_end - layer_c_start)
    print(f"Layer C processing time (s): {layer_c_processing_time_s}")
    
    return flagged_texts, flagged_labels

def evaluate_llm_judge(texts, labels):
    from core.layer_e.llm_judge import call_judge
    
    layer_e_start = time.time()
    results = []
    for idx, text in enumerate(texts):
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(texts)}...")
        
        result = call_judge(text)
        if result is None:
            # Skip if LLM call failed
            print("Result was None")
            continue
            
        predicted_label = result.label 
        results.append({
            'text': text[:200],  # Store first 200 chars for debugging
            'true_label': labels[idx],
            'llm_label': result.label,
            'predicted_label': predicted_label,
            'rationale': result.rationale,
        })
    layer_e_end = time.time()
    layer_e_processing_time_s = (layer_e_end - layer_e_start)
    print(f"Layer E processing time (s): {layer_e_processing_time_s}")
    
    results_df = pd.DataFrame(results)
    
    # Show misclassified samples (false negatives - malicious classified as benign)
    # false_negatives = results_df[(results_df['true_label'] == 1) & (results_df['llm_label'] == 0)]
    # if len(false_negatives) > 0:
    #     print("\n" + "="*60)
    #     print("FALSE NEGATIVES (Malicious → Benign) - MISSED ATTACKS")
    #     print("="*60)
    #     for idx, row in false_negatives.iterrows():
    #         print(f"\n--- Sample {idx} ---")
    #         print(f"Text: {row['text'][:300]}...")
    #         print(f"Rationale: {row['rationale']}")
    
    # Confusion Matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    
    safe_benign = ((results_df['true_label'] == 0) & (results_df['llm_label'] == 0)).sum()
    safe_malicious = ((results_df['true_label'] == 0) & (results_df['llm_label'] == 1)).sum()
    
    malicious_benign = ((results_df['true_label'] == 1) & (results_df['llm_label'] == 0)).sum()
    malicious_malicious = ((results_df['true_label'] == 1) & (results_df['llm_label'] == 1)).sum()
    
    safe_total = safe_benign + safe_malicious
    malicious_total = malicious_benign + malicious_malicious
    total = safe_total + malicious_total
    
    print(f"{'Ground Truth':<15} | {'Benign':>10} | {'Malicious':>10} | {'Total':>10}")
    print("-" * 60)
    print(f"{'SAFE':<15} | {safe_benign:>10} | {safe_malicious:>10} | {safe_total:>10}")
    print(f"{'MALICIOUS':<15} | {malicious_benign:>10} | {malicious_malicious:>10} | {malicious_total:>10}")
    print("-" * 60)
    print(f"{'Total':<15} | {safe_benign + malicious_benign:>10} | {safe_malicious + malicious_malicious:>10} | {total:>10}")
    print("="*60)
    
    tp = ((results_df['predicted_label'] == 1) & (results_df['true_label'] == 1)).sum()
    fp = ((results_df['predicted_label'] == 1) & (results_df['true_label'] == 0)).sum()
    fn = ((results_df['predicted_label'] == 0) & (results_df['true_label'] == 1)).sum()
    
    accuracy = (results_df['predicted_label'] == results_df['true_label']).mean()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

def test_layer_e():
    test_texts, true_labels = load_test_data("datasets/barrikada_test.csv")

    # Filter through Layer B
    texts_after_b, labels_after_b = filter_through_layer_b(test_texts, true_labels)
    print(f"After Layer B: {len(texts_after_b)} samples remain.")

    # Filter through Layer C
    texts_after_c, labels_after_c = filter_through_layer_c(texts_after_b, labels_after_b)
    print(f"After Layer C: {len(texts_after_c)} samples remain.")
    
    evaluate_llm_judge(texts_after_c, labels_after_c)
    
if __name__ == "__main__":
    start_time = time.time()
    test_layer_e()
    end_time = time.time()
    print(f"Total Execution time: {end_time - start_time}s")