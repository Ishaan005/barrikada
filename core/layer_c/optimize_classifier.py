"""
Optimize classifier thresholds and model configurations
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from classifier import Classifier
from core.settings import Settings

settings = Settings()


def evaluate_classifier(classifier, test_df):
    """Evaluate classifier performance"""
    results = []
    for _, row in test_df.iterrows():
        result = classifier.predict(row['text'])
        results.append({
            'true_label': row['label'],
            'verdict': result.verdict,
            'probability_score': result.probability_score
        })
    
    results_df = pd.DataFrame(results)
    verdict_map = {'allow': 0, 'flag': 1, 'block': 1}
    results_df['predicted_label'] = results_df['verdict'].map(verdict_map)
    
    tp = len(results_df[(results_df['true_label'] == 1) & (results_df['predicted_label'] == 1)])
    tn = len(results_df[(results_df['true_label'] == 0) & (results_df['predicted_label'] == 0)])
    fp = len(results_df[(results_df['true_label'] == 0) & (results_df['predicted_label'] == 1)])
    fn = len(results_df[(results_df['true_label'] == 1) & (results_df['predicted_label'] == 0)])
    
    accuracy = (tp + tn) / len(results_df)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fp': fp,
        'fn': fn,
    }


def test_thresholds():
    """Test different threshold configurations"""
    print("Testing threshold configurations...")
    
    test_df = pd.read_csv(settings.dataset_path.replace('barrikada.csv', 'barrikada_test.csv'))
    print(f"Test set: {len(test_df)} samples ({len(test_df[test_df['label']==0])} safe, {len(test_df[test_df['label']==1])} malicious)\n")
    
    configs = [
        {'name': 'Current', 'low': 0.25, 'high': 0.75},
        {'name': 'Fewer FP', 'low': 0.35, 'high': 0.80},
        {'name': 'Alt 1', 'low': 0.30, 'high': 0.80},
        {'name': 'Alt 2', 'low': 0.40, 'high': 0.85},
        {'name': 'Narrow FLAG', 'low': 0.20, 'high': 0.85},
        {'name': 'Balanced', 'low': 0.30, 'high': 0.75},
    ]
    
    results = []
    
    for config in configs:
        classifier = Classifier(
            vectorizer_path=settings.vectorizer_path,
            model_path=settings.model_path,
            low=config['low'],
            high=config['high']
        )
        
        metrics = evaluate_classifier(classifier, test_df)
        
        print(f"{config['name']:15} (low={config['low']}, high={config['high']}): "
              f"Acc={metrics['accuracy']*100:5.1f}% "
              f"Prec={metrics['precision']*100:5.1f}% "
              f"Rec={metrics['recall']*100:5.1f}% "
              f"F1={metrics['f1_score']:.3f} "
              f"FP={metrics['fp']:3} FN={metrics['fn']:3}")
        
        results.append({
            'config': config['name'],
            'low': config['low'],
            'high': config['high'],
            **metrics
        })
    
    df = pd.DataFrame(results)
    
    best_f1 = df.loc[df['f1_score'].idxmax()]
    best_acc = df.loc[df['accuracy'].idxmax()]
    
    print(f"\nBest F1: {best_f1['config']} (low={best_f1['low']}, high={best_f1['high']}) - F1={best_f1['f1_score']:.3f}")
    print(f"Best Acc: {best_acc['config']} (low={best_acc['low']}, high={best_acc['high']}) - Acc={best_acc['accuracy']*100:.1f}%")
    
    output = Path(__file__).parent / "models" / "threshold_results.csv"
    df.to_csv(output, index=False)
    print(f"\nSaved to {output}")
    
    return df


def test_models():
    """Test different model configurations"""
    print("\nTesting model configurations...")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import joblib
    import time
    
    df = pd.read_csv(settings.dataset_path)
    X = df['text'].values
    y = df['label'].astype(int).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) #type:ignore
    
    configs = [
        {
            'name': 'Baseline',
            'vec': TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=5000),
            'clf': LinearSVC(class_weight='balanced', dual=False, max_iter=2000, random_state=42)
        },
        {
            'name': '10k features',
            'vec': TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=10000),
            'clf': LinearSVC(class_weight='balanced', dual=False, max_iter=2000, random_state=42)
        },
        {
            'name': 'Stopwords',
            'vec': TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=5000, stop_words='english'),
            'clf': LinearSVC(class_weight='balanced', dual=False, max_iter=2000, random_state=42)
        },
        {
            'name': 'Char ngrams',
            'vec': TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=5000),
            'clf': LinearSVC(class_weight='balanced', dual=False, max_iter=2000, random_state=42)
        },
        {
            'name': 'Higher weight',
            'vec': TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=5000),
            'clf': LinearSVC(class_weight={0: 1, 1: 3}, dual=False, max_iter=2000, random_state=42)
        },
        {
            'name': 'Combined',
            'vec': TfidfVectorizer(analyzer="word", ngram_range=(1,3), max_features=10000, stop_words='english'),
            'clf': LinearSVC(class_weight={0: 1, 1: 2}, dual=False, max_iter=2000, random_state=42)
        }
    ]
    
    results = []
    
    for config in configs:
        pipeline = Pipeline([('tfidf', config['vec']), ('clf', config['clf'])])
        
        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start
        
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.decision_function(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)
        
        print(f"{config['name']:15} Acc={acc*100:5.1f}% Prec={prec*100:5.1f}% Rec={rec*100:5.1f}% F1={f1:.3f} AUC={auc:.3f} ({train_time:.1f}s)")
        
        results.append({
            'config': config['name'],
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc,
            'train_time': train_time
        })
        
        # Save if good
        if f1 > 0.95:
            model_path = settings.model_path.replace('.joblib', f'_{config["name"].replace(" ", "_")}.joblib')
            vec_path = settings.vectorizer_path.replace('.joblib', f'_{config["name"].replace(" ", "_")}.joblib')
            joblib.dump(pipeline.named_steps['clf'], model_path)
            joblib.dump(pipeline.named_steps['tfidf'], vec_path)
            print(f"  -> Saved (F1={f1:.3f})")
    
    df = pd.DataFrame(results)
    best = df.loc[df['f1_score'].idxmax()]
    
    print(f"\nBest: {best['config']} - F1={best['f1_score']:.3f}, Acc={best['accuracy']*100:.1f}%")
    
    output = Path(__file__).parent / "models" / "model_results.csv"
    df.to_csv(output, index=False)
    print(f"Saved to {output}")
    
    return df


if __name__ == "__main__":
    print("Classifier Optimization\n")
    
    # Quick threshold test
    threshold_df = test_thresholds()
    
    # Model test
    print("\n" + "="*60)
    choice = input("Test model configurations? (y/n): ")
    
    if choice.lower() == 'y':
        model_df = test_models()
        
        print("\n" + "="*60)
        print("To apply best thresholds, update classifier.py:")
        best = threshold_df.loc[threshold_df['f1_score'].idxmax()]
        print(f"  self.low_threshold = {best['low']}")
        print(f"  self.high_threshold = {best['high']}")
    else:
        print("\nDone. Apply recommended thresholds in classifier.py")

