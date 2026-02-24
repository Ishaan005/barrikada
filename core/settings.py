from pydantic import BaseModel
from pathlib import Path


class Settings(BaseModel):
    app_name: str = "Barrikada"
    debug_mode: bool = False

    # Use absolute paths based on project root
    _project_root: Path = Path(__file__).parent.parent

    ### Layer B (signatures)
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
    

    ### Layer C
    # Routing thresholds for Layer C classifier:
    layer_c_low_threshold: float = 0.25
    layer_c_high_threshold: float = 0.75

    layer_c_seed: int = 42

    layer_c_val_test_size: float = 0.30
    layer_c_test_split: float = 0.50

    layer_c_embedding_model: str = "all-mpnet-base-v2"
    layer_c_embedding_batch_size: int = 128

    layer_c_xgb_n_estimators: int = 2000
    layer_c_xgb_max_depth: int = 7
    layer_c_xgb_learning_rate: float = 0.05
    layer_c_xgb_subsample: float = 0.8
    layer_c_xgb_colsample_bytree: float = 0.9
    layer_c_xgb_early_stopping_rounds: int = 80
    layer_c_xgb_tree_method: str = "hist"
    layer_c_xgb_reg_alpha: float = 0.1      # L1 regularization
    layer_c_xgb_reg_lambda: float = 1.0     # L2 regularization
    layer_c_xgb_min_child_weight: int = 5   # min samples per leaf
    layer_c_xgb_gamma: float = 0.1          # min split gain

    # Threshold tuning constraints
    layer_c_tune_target_block_precision: float = 0.99
    layer_c_tune_max_malicious_allow_rate: float = 0.05
    layer_c_tune_max_safe_fpr: float = 0.10          # max 10% of safe texts flagged+blocked
    layer_c_tune_min_flag_band: float = 0.05

    # Threshold grid search
    layer_c_tune_low_grid_start: float = 0.25
    layer_c_tune_low_grid_end: float = 0.60
    layer_c_tune_high_grid_start: float = 0.55
    layer_c_tune_high_grid_end: float = 0.99
    layer_c_tune_grid_steps: int = 200

    # Composite score weights for threshold selection
    layer_c_tune_w_block_recall: float = 0.4
    layer_c_tune_w_allow_rate: float = 0.3
    layer_c_tune_w_flag_rate: float = 0.3

    @property
    def dataset_path(self) -> str:
        return str(self._project_root / "datasets" / "barrikada.csv")
    
    @property
    def model_path(self) -> str:
        return str(self._project_root / "core/layer_c" / "outputs" / "classifier.joblib")

    @property
    def meta_features_path(self) -> str:
        return str(self._project_root / "core/layer_c" / "outputs" / "meta_features.joblib")

