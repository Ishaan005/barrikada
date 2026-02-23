from typing import Tuple
from pydantic import BaseModel
from pathlib import Path


class Settings(BaseModel):
    app_name: str = "Barrikada"
    debug_mode: bool = False

    # Use absolute paths based on project root
    _project_root: Path = Path(__file__).parent.parent

    # ----- Layer B (signatures) -----
    # Scoring/thresholds
    layer_b_allow_confidence: float = 0.99
    layer_b_block_min_hits: int = 1
    layer_b_block_min_rule_precision: float = 0.95
    layer_b_flag_confidence: float = 0.85
    layer_b_block_confidence: float = 0.95

    @property
    def layer_b_signatures_extracted_dir(self) -> str:
        return str(self._project_root / "core/layer_b" / "signatures" / "extracted")

    @property
    def layer_b_malicious_rules_path(self) -> str:
        return str(Path(self.layer_b_signatures_extracted_dir) / "malicious_block_high_signatures.yar")

    @property
    def layer_b_allow_rules_path(self) -> str:
        return str(Path(self.layer_b_signatures_extracted_dir) / "safe_allow_signatures.yar")
    

    # ----- Layer C -----
    # Routing thresholds for Layer C classifier:
    # score < low => allow, low <= score < high => flag, score >= high => block
    layer_c_low_threshold: float = 0.25
    layer_c_high_threshold: float = 0.75

    # Reproducibility
    layer_c_seed: int = 42

    # Train / val / test split
    # 70% train, 15% val, 15% test (first split takes 30%, second splits that 50/50)
    layer_c_val_test_size: float = 0.30
    layer_c_test_split: float = 0.50

    # TF-IDF vectorizer
    layer_c_tfidf_word_max_features: int = 80_000
    layer_c_tfidf_char_max_features: int = 80_000
    layer_c_tfidf_min_df: int = 5
    layer_c_tfidf_max_df: float = 0.95
    layer_c_tfidf_word_ngram_range: Tuple[int, int] = (1, 2)
    layer_c_tfidf_char_ngram_range: Tuple[int, int] = (3, 5)

    # SVD dimensionality reduction
    layer_c_svd_components: int = 500

    # XGBoost hyperparameters
    layer_c_xgb_n_estimators: int = 1500
    layer_c_xgb_max_depth: int = 8
    layer_c_xgb_learning_rate: float = 0.05
    layer_c_xgb_subsample: float = 0.8
    layer_c_xgb_colsample_bytree: float = 0.8
    layer_c_xgb_early_stopping_rounds: int = 50

    # Threshold tuning constraints
    layer_c_tune_target_block_precision: float = 0.99
    layer_c_tune_max_malicious_allow_rate: float = 0.02
    layer_c_tune_min_flag_band: float = 0.05

    # Threshold grid search bounds and resolution
    layer_c_tune_low_grid_start: float = 0.05
    layer_c_tune_low_grid_end: float = 0.60
    layer_c_tune_high_grid_start: float = 0.40
    layer_c_tune_high_grid_end: float = 0.99
    layer_c_tune_grid_steps: int = 200

    # Composite score weights for threshold selection
    # score = w_recall * block_recall + w_allow * allow_rate - w_flag * flag_rate
    layer_c_tune_w_block_recall: float = 0.5
    layer_c_tune_w_allow_rate: float = 0.3
    layer_c_tune_w_flag_rate: float = 0.2

    @property
    def dataset_path(self) -> str:
        return str(self._project_root / "datasets" / "barrikada.csv")
    
    @property
    def model_path(self) -> str:
        return str(self._project_root / "core/layer_c" / "outputs" / "classifier.joblib")
    
    @property
    def vectorizer_path(self) -> str:
        return str(self._project_root / "core/layer_c" / "outputs" / "tf_idf_vectorizer.joblib")

    @property
    def reducer_path(self) -> str:
        return str(self._project_root / "core/layer_c" / "outputs" / "pca_reducer.joblib")

