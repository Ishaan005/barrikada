import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
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
            'layer_b_score': layer_b_result.total_score,
            'layer_b_confidence': layer_b_result.confidence_score,
            'predicted_label': predicted_label,
            'is_correct': is_correct,
            'highest_severity': layer_b_result.highest_severity.value if layer_b_result.highest_severity else None,
            'processing_time_ms': layer_b_result.processing_time_ms,
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"core/layer_b/outputs/layer_b_results_{timestamp}.csv"
    
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Quick stats
    results_df = pd.DataFrame(results)
    
    accuracy = results_df['is_correct'].mean()
    tp = ((results_df['predicted_label'] == 1) & (results_df['true_label'] == 1)).sum()
    fp = ((results_df['predicted_label'] == 1) & (results_df['true_label'] == 0)).sum()
    fn = ((results_df['predicted_label'] == 0) & (results_df['true_label'] == 1)).sum()

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Recall: {tp /(tp+fn)}")
    print(f"Total detections: {results_df['predicted_label'].sum()}")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    
    return output_path

if __name__ == "__main__":
    test_layer_b()
