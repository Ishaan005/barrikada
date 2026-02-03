import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import sys
import io
from contextlib import redirect_stdout

from core.layer_a.pipeline import analyze_text
from core.layer_b.signature_engine import SignatureEngine

def test_layer_b():
    
    df = pd.read_csv("datasets/barrikada_test.csv")
    
    print(f"Testing Layer B on {len(df)} samples...")
    layer_b = SignatureEngine()
    
    results = []
    
    for idx in range(len(df)):
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(df)}...")
        
        row = df.iloc[idx]
        
        # supress Layer A output
        with redirect_stdout(io.StringIO()):
            layer_a_result = analyze_text(row['text'])
        
        # Run Layer B on preprocessed text
        layer_b_result = layer_b.detect(layer_a_result.processed_text)

        if layer_b_result.verdict == "block":
            predicted_label = 1
        elif layer_b_result.verdict == "flag":
            predicted_label = row['label']  # keep ground truth for "flag"
        else:
            predicted_label = 0
        
        # Correct if predicted matches ground truth
        is_correct = predicted_label == row['label']
        
        results.append({
            'text': row['text'],
            'preprocessed_text': layer_a_result.processed_text,
            'true_label': row['label'],
            'layer_b_matches': len(layer_b_result.matches),
            'layer_b_verdict': layer_b_result.verdict,
            'layer_b_confidence': layer_b_result.confidence_score,
            'predicted_label': predicted_label,
            'is_correct': is_correct,
            'allowlisted': getattr(layer_b_result, 'allowlisted', False),
            'processing_time_ms': layer_b_result.processing_time_ms,
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"core/layer_b/outputs/layer_b_results_{timestamp}.csv"
    
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Quick stats
    results_df = pd.DataFrame(results)

    # Verdict breakdown
    verdict_counts = (
        results_df["layer_b_verdict"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .value_counts()
    )
    total = verdict_counts.sum()

    
    # Confusion Matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    
    # Calculate confusion matrix values
    safe_allow = ((results_df['true_label'] == 0) & (results_df['layer_b_verdict'] == 'allow')).sum()
    safe_flag = ((results_df['true_label'] == 0) & (results_df['layer_b_verdict'] == 'flag')).sum()
    safe_block = ((results_df['true_label'] == 0) & (results_df['layer_b_verdict'] == 'block')).sum()
    
    malicious_allow = ((results_df['true_label'] == 1) & (results_df['layer_b_verdict'] == 'allow')).sum()
    malicious_flag = ((results_df['true_label'] == 1) & (results_df['layer_b_verdict'] == 'flag')).sum()
    malicious_block = ((results_df['true_label'] == 1) & (results_df['layer_b_verdict'] == 'block')).sum()
    
    # Calculate totals
    safe_total = safe_allow + safe_flag + safe_block
    malicious_total = malicious_allow + malicious_flag + malicious_block
    
    # Print table header
    print(f"{'Ground Truth':<15} | {'Allow':>10} | {'Flag':>10} | {'Block':>10} | {'Total':>10}")
    print("-" * 60)
    
    # Print SAFE row
    print(f"{'SAFE':<15} | {safe_allow:>10} | {safe_flag:>10} | {safe_block:>10} | {safe_total:>10}")
    
    # Print MALICIOUS row
    print(f"{'MALICIOUS':<15} | {malicious_allow:>10} | {malicious_flag:>10} | {malicious_block:>10} | {malicious_total:>10}")
    
    # Print total row
    allow_total = safe_allow + malicious_allow
    flag_total = safe_flag + malicious_flag
    block_total = safe_block + malicious_block
    print("-" * 60)
    print(f"{'Total':<15} | {allow_total:>10} | {flag_total:>10} | {block_total:>10} | {total:>10}")
    print("="*60)
    
    tp = ((results_df['predicted_label'] == 1) & (results_df['true_label'] == 1)).sum()
    fn = ((results_df['predicted_label'] == 0) & (results_df['true_label'] == 1)).sum()

    accuracy = (results_df['predicted_label'] == results_df['true_label']).mean()

    print(f"Accuracy: {accuracy}")
    print(f"Recall: {tp /(tp+fn)}")
    print(f"Total detections: {results_df['predicted_label'].sum()}")
    
    return output_path

if __name__ == "__main__":
    import time
    start_time = time.time()
    test_layer_b()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}s")