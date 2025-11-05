import sys
from pathlib import Path

# Add project root to path for imports
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline

from core.settings import Settings
from models.LayerCResult import LayerCResult

settings = Settings()

INPUT_PATH = settings.dataset_path
MODEL_PATH = settings.model_path
VECTORIZER_PATH = settings.vectorizer_path

class Classifier:
    def __init__(self, vectorizer_path, model_path, low = 0.25, high = 0.75):
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        self.low_threshold = low
        self.high_threshold = high
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=5000)),
            ('clf', LinearSVC(class_weight='balanced', dual=False, max_iter=2000))
        ])

    def __load_data(self, file_path):
        df = pd.read_csv(file_path)
        df['text'] = df['text']
        X = df['text'].values
        y = df['label'].astype(int).values
        return X, y
    
    def train(self):
        X, y = self.__load_data(INPUT_PATH) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=np.array(y), random_state=42)
        
        start_time = time.time()

        self.pipeline.fit(X_train, y_train)
        vectorizer = self.pipeline.named_steps['tfidf']
        clf = self.pipeline.named_steps['clf']

        train_time = time.time() - start_time

        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.decision_function(X_test)
        print(classification_report(y_test, y_pred))
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        print(f"Training time: {train_time:.2f} seconds")

        joblib.dump(clf, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print("Model and vectorizer saved.")

    def predict(self, input_text: str) -> LayerCResult:
        """
        Predict on input text and return standardized result
        
        Args:
            input_text: Text to classify
            
        Returns:
            LayerCResult: Standardized result object
        """
        start_time = time.time()
        
        # Vectorize and predict
        vec = self.vectorizer.transform([input_text])
        probability_score = float(self.model.predict_proba(vec)[0, 1])

        # Determine verdict based on thresholds
        if probability_score < self.low_threshold:
            verdict = 'allow'
            confidence_score = 1.0 - probability_score  # Higher confidence for lower scores
        elif self.low_threshold <= probability_score < self.high_threshold:
            verdict = 'flag'
            # Confidence is lower in the uncertain middle range
            distance_from_center = abs(probability_score - 0.5)
            confidence_score = 0.5 + (distance_from_center * 0.4)  # 0.5 to 0.7 range
        else:
            verdict = 'block'
            confidence_score = probability_score  # Higher confidence for higher scores
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return LayerCResult(
            verdict=verdict,
            probability_score=probability_score,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms
        )


if __name__ == "__main__":
    """
    Run training when this file is executed directly
    Usage: python classifier.py
    """
    print("=" * 60)
    print("Starting Classifier Training")
    print("=" * 60)
    
    settings = Settings()
    
    print(f"\nDataset: {settings.dataset_path}")
    print(f"Model output: {settings.model_path}")
    print(f"Vectorizer output: {settings.vectorizer_path}\n")
    
    # Create a minimal classifier instance just for training
    classifier = object.__new__(Classifier)
    classifier.pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=5000)),
        ('clf', LinearSVC(class_weight='balanced', dual=False, max_iter=2000))
    ])
    
    # Call the train method
    classifier.train()
