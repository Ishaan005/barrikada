"""
Plotting functions for classifier analysis
"""
import matplotlib.pyplot as plt
from pathlib import Path    
import seaborn as sns
import numpy as np
import math
import pandas as pd
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve, 
                            average_precision_score, precision_score, recall_score, f1_score)


def plot_training_results(train_results):
    """Plot training results: confusion matrix, ROC, PR curve, thresholds"""
    y_test = train_results['y_test']
    y_pred = train_results['y_pred']
    y_proba = train_results['y_proba']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Results', fontsize=16, fontweight='bold')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Safe', 'Malicious'], yticklabels=['Safe', 'Malicious'])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True')
    axes[0, 0].set_xlabel('Predicted')
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    axes[0, 0].text(0.5, -0.15, f'Accuracy: {accuracy:.2%}', 
                   transform=axes[0, 0].transAxes, ha='center', fontsize=12)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0, 1].set_xlabel('FPR')
    axes[0, 1].set_ylabel('TPR')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(alpha=0.3)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    axes[0, 2].plot(recall, precision, color='green', lw=2, label=f'AP = {avg_precision:.3f}')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision-Recall')
    axes[0, 2].legend(loc="lower left")
    axes[0, 2].grid(alpha=0.3)
    
    # Score Distribution
    proba_safe = [1.0 / (1.0 + math.exp(-s)) for s, l in zip(y_proba, y_test) if l == 0]
    proba_mal = [1.0 / (1.0 + math.exp(-s)) for s, l in zip(y_proba, y_test) if l == 1]
    
    axes[1, 0].hist(proba_safe, bins=50, alpha=0.5, label='Safe', color='blue', density=True)
    axes[1, 0].hist(proba_mal, bins=50, alpha=0.5, label='Malicious', color='red', density=True)
    axes[1, 0].axvline(x=0.25, color='orange', linestyle='--', label='Low (0.25)')
    axes[1, 0].axvline(x=0.75, color='darkred', linestyle='--', label='High (0.75)')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Metrics Bar Chart
    metrics = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'Acc': accuracy
    }
    bars = axes[1, 1].bar(metrics.keys(), metrics.values(), 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].set_title('Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    for bar in bars:
        h = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., h, f'{h:.3f}',
                       ha='center', va='bottom', fontweight='bold')
    
    # Threshold Analysis
    thresholds = np.linspace(0, 1, 100)
    precs, recs, f1s = [], [], []
    
    for thresh in thresholds:
        y_pred_t = [1 if (1.0 / (1.0 + math.exp(-s))) >= thresh else 0 for s in y_proba]
        if sum(y_pred_t) > 0:
            precs.append(precision_score(y_test, y_pred_t, zero_division=0))
            recs.append(recall_score(y_test, y_pred_t, zero_division=0))
            f1s.append(f1_score(y_test, y_pred_t, zero_division=0))
        else:
            precs.append(0)
            recs.append(0)
            f1s.append(0)
    
    axes[1, 2].plot(thresholds, precs, label='Precision', lw=2)
    axes[1, 2].plot(thresholds, recs, label='Recall', lw=2)
    axes[1, 2].plot(thresholds, f1s, label='F1', lw=2)
    axes[1, 2].axvline(x=0.25, color='orange', linestyle='--', alpha=0.7)
    axes[1, 2].axvline(x=0.75, color='darkred', linestyle='--', alpha=0.7)
    axes[1, 2].set_xlabel('Threshold')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Threshold Impact')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output = Path(__file__).parent / "models" / "training_analysis.png"
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output}")
    plt.show()
    
    return metrics


def plot_test_results(test_df):
    """Plot test set results: verdicts, probabilities, confidence"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Test Results', fontsize=16, fontweight='bold')
    
    verdict_map = {'allow': 0, 'flag': 1, 'block': 1}
    test_df['predicted_label'] = test_df['verdict'].map(verdict_map)
    
    # Verdict Distribution
    counts = test_df['verdict'].value_counts()
    colors = {'allow': '#96CEB4', 'flag': '#FFD93D', 'block': '#FF6B6B'}
    axes[0, 0].bar(counts.index, counts.values, 
                  color=[colors.get(v, 'gray') for v in counts.index], alpha=0.8)
    axes[0, 0].set_title('Verdicts')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    for i, (v, c) in enumerate(counts.items()):
        axes[0, 0].text(i, c, str(c), ha='center', va='bottom', fontweight='bold')
    
    # Probability by Label
    safe = test_df[test_df['true_label'] == 0]['probability_score']
    mal = test_df[test_df['true_label'] == 1]['probability_score']
    
    axes[0, 1].hist(safe, bins=30, alpha=0.5, label='Safe', color='blue', density=True)
    axes[0, 1].hist(mal, bins=30, alpha=0.5, label='Malicious', color='red', density=True)
    axes[0, 1].axvline(x=0.25, color='orange', linestyle='--', label='Low')
    axes[0, 1].axvline(x=0.75, color='darkred', linestyle='--', label='High')
    axes[0, 1].set_xlabel('Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Probability by Label')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Confidence Distribution
    axes[1, 0].hist(test_df['confidence_score'], bins=30, color='purple', alpha=0.7)
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Confidence')
    axes[1, 0].axvline(x=test_df['confidence_score'].mean(), color='red', 
                      linestyle='--', label=f"Mean: {test_df['confidence_score'].mean():.3f}")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Summary Table
    correct = (
        ((test_df['verdict'] == 'allow') & (test_df['true_label'] == 0)) |
        ((test_df['verdict'].isin(['flag', 'block'])) & (test_df['true_label'] == 1))
    ).sum()
    
    total = len(test_df)
    acc = correct / total
    
    axes[1, 1].axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['Total', str(total)],
        ['Correct', str(correct)],
        ['Incorrect', str(total - correct)],
        ['Accuracy', f"{acc:.2%}"],
        ['', ''],
        ['Allow', str((test_df['verdict'] == 'allow').sum())],
        ['Flag', str((test_df['verdict'] == 'flag').sum())],
        ['Block', str((test_df['verdict'] == 'block').sum())],
    ]
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='left', loc='center',
                            colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    table[(0, 0)].set_facecolor('#4ECDC4')
    table[(0, 1)].set_facecolor('#4ECDC4')
    table[(0, 0)].set_text_props(weight='bold', color='white')
    table[(0, 1)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Summary', pad=20, fontweight='bold')
    
    plt.tight_layout()
    
    output = Path(__file__).parent / "models" / "test_analysis.png"
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.show()
    
    return {'accuracy': acc, 'total': total, 'correct': correct}


def analyze_errors(classifier, test_df):
    """Analyze errors and provide recommendations"""
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    verdict_map = {'allow': 0, 'flag': 1, 'block': 1}
    test_df['predicted_label'] = test_df['verdict'].map(verdict_map)
    
    # False Positives: Safe classified as malicious
    fp_df = test_df[(test_df['true_label'] == 0) & (test_df['predicted_label'] == 1)]
    
    # False Negatives: Malicious classified as safe  
    fn_df = test_df[(test_df['true_label'] == 1) & (test_df['predicted_label'] == 0)]
    
    tp = len(test_df[(test_df['true_label'] == 1) & (test_df['predicted_label'] == 1)])
    tn = len(test_df[(test_df['true_label'] == 0) & (test_df['predicted_label'] == 0)])
    fp = len(fp_df)
    fn = len(fn_df)
    total = len(test_df)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPerformance:")
    print(f"  Accuracy: {((tp+tn)/total)*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall: {recall*100:.2f}%")
    print(f"  F1: {f1:.3f}")
    print(f"\n  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    # Verdict breakdown
    allow_safe = len(test_df[(test_df['verdict'] == 'allow') & (test_df['true_label'] == 0)])
    allow_mal = len(test_df[(test_df['verdict'] == 'allow') & (test_df['true_label'] == 1)])
    flag_safe = len(test_df[(test_df['verdict'] == 'flag') & (test_df['true_label'] == 0)])
    flag_mal = len(test_df[(test_df['verdict'] == 'flag') & (test_df['true_label'] == 1)])
    block_safe = len(test_df[(test_df['verdict'] == 'block') & (test_df['true_label'] == 0)])
    block_mal = len(test_df[(test_df['verdict'] == 'block') & (test_df['true_label'] == 1)])
    
    print(f"\nVerdicts:")
    print(f"  ALLOW: {allow_safe} safe, {allow_mal} malicious")
    print(f"  FLAG: {flag_safe} safe, {flag_mal} malicious")
    print(f"  BLOCK: {block_safe} safe, {block_mal} malicious")
    
    # Error examples
    if len(fn_df) > 0:
        print(f"\nFalse Negatives ({fn}):")
        avg_fn = fn_df['probability_score'].mean()
        print(f"  Avg probability: {avg_fn:.4f}")
        for _, row in fn_df.head(3).iterrows():
            print(f"    [{row['probability_score']:.4f}] {row['text'][:80]}...")
    
    if len(fp_df) > 0:
        print(f"\nFalse Positives ({fp}):")
        avg_fp = fp_df['probability_score'].mean()
        print(f"  Avg probability: {avg_fp:.4f}")
        for _, row in fp_df.head(3).iterrows():
            print(f"    [{row['probability_score']:.4f}] {row['text'][:80]}...")
    
    # Recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    recs = []
    
    if fn > fp * 2:
        recs.append({
            'priority': 'HIGH',
            'issue': f'High FN rate ({fn} missed)',
            'action': f'Lower LOW threshold: {classifier.low_threshold} → 0.18',
            'impact': 'Catch more attacks, slight FP increase'
        })
    
    if fp > fn * 2:
        recs.append({
            'priority': 'HIGH',
            'issue': f'High FP rate ({fp} false alarms)',
            'action': f'Raise LOW threshold: {classifier.low_threshold} → 0.35',
            'impact': 'Reduce false alarms, may miss some attacks'
        })
    
    flag_acc = flag_mal / (flag_safe + flag_mal) if (flag_safe + flag_mal) > 0 else 0
    if flag_acc < 0.6:
        recs.append({
            'priority': 'MEDIUM',
            'issue': f'FLAG zone low accuracy ({flag_acc*100:.1f}%)',
            'action': 'Narrow FLAG zone: try low=0.20, high=0.85',
            'impact': 'More decisive classifications'
        })
    
    if (tp + tn) / total < 0.85:
        recs.append({
            'priority': 'HIGH',
            'issue': f'Accuracy below 85% ({((tp+tn)/total)*100:.1f}%)',
            'action': 'Improve model: more data, tune features, try ensemble',
            'impact': 'Better overall detection'
        })
    
    safe_count = len(test_df[test_df['true_label'] == 0])
    mal_count = len(test_df[test_df['true_label'] == 1])
    imbalance = max(safe_count, mal_count) / min(safe_count, mal_count)
    
    if imbalance > 3:
        recs.append({
            'priority': 'MEDIUM',
            'issue': f'Class imbalance: {safe_count}:{mal_count} ({imbalance:.1f}:1)',
            'action': f'Add class_weight={{0:1, 1:{int(imbalance)}}} to model',
            'impact': 'Better precision/recall balance'
        })
    
    if recall < 0.8:
        recs.append({
            'priority': 'MEDIUM',
            'issue': f'Low recall ({recall*100:.1f}%)',
            'action': 'Add features: char n-grams, pattern counts, structure',
            'impact': 'Catch more attack patterns'
        })
    
    for i, r in enumerate(recs, 1):
        print(f"\n{i}. [{r['priority']}] {r['issue']}")
        print(f"   Action: {r['action']}")
        print(f"   Impact: {r['impact']}")
    
    # Save errors
    if len(fp_df) + len(fn_df) > 0:
        errors = pd.concat([
            fn_df.assign(error_type='false_negative'),
            fp_df.assign(error_type='false_positive')
        ])
        output = Path(__file__).parent / "models" / "misclassifications.csv"
        errors.to_csv(output, index=False)
        print(f"\nSaved {len(errors)} misclassifications to: {output}")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'accuracy': (tp + tn) / total,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_positives': fp,
        'false_negatives': fn,
        'recommendations': recs
    }

