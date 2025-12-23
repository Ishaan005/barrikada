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
    
    df = pd.read_csv("datasets/barrikada.csv")
    
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

        if layer_b_result.verdict in ["block", "flag"]:
            predicted_label = 1
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
    total = int(verdict_counts.sum())
    safe_count = int(verdict_counts.get("allow", 0))
    flag_count = int(verdict_counts.get("flag", 0))
    block_count = int(verdict_counts.get("block", 0))

    print("\nVerdict breakdown:")
    print(f"  allow:  {safe_count} ({safe_count/total:.1%})")
    print(f"  flag:  {flag_count} ({flag_count/total:.1%})")
    print(f"  block: {block_count} ({block_count/total:.1%})")
    
    accuracy = results_df['is_correct'].mean()
    tp = ((results_df['predicted_label'] == 1) & (results_df['true_label'] == 1)).sum()
    fp = ((results_df['predicted_label'] == 1) & (results_df['true_label'] == 0)).sum()
    fn = ((results_df['predicted_label'] == 0) & (results_df['true_label'] == 1)).sum()

    # False negatives breakdown
    fn_df = results_df[(results_df['predicted_label'] == 1) & (results_df['true_label'] == 0)]
    if len(fn_df) > 0:
        fn_verdict_counts = (
            fn_df["layer_b_verdict"]
            .value_counts()
        )
        fn_total = int(fn_verdict_counts.sum())

        print("\nFalse negatives breakdown")
        for verdict, count in fn_verdict_counts.items():
            count_int = int(count)
            print(f"  {verdict}: {count_int} ({count_int/fn_total:.1%})")

        if "allowlisted" in fn_df.columns:
            fn_allowlisted = int(fn_df["allowlisted"].fillna(False).astype(bool).sum())
            print(f"  allowlisted_in_fn: {fn_allowlisted} ({fn_allowlisted/fn_total:.1%})")

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Recall: {tp /(tp+fn)}")
    print(f"Total detections: {results_df['predicted_label'].sum()}")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    
    return output_path

if __name__ == "__main__":
    test_layer_b()
