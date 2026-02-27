import torch
from core.settings import Settings
from xgboost import XGBClassifier

_settings = Settings()
SEED = _settings.layer_c_seed
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # layer_c_xgb_n_estimators: int = 3000
    # layer_c_xgb_max_depth: int = 7
    # layer_c_xgb_learning_rate: float = 0.05
    # layer_c_xgb_subsample: float = 0.8
    # layer_c_xgb_colsample_bytree: float = 0.9
    # layer_c_xgb_scale_pos_multiplier: float = 1.5  # extra FN penalty on top of base class ratio
    # layer_c_xgb_early_stopping_rounds: int = 150
    # layer_c_xgb_tree_method: str = "hist"
    # layer_c_xgb_reg_alpha: float = 0.1      # L1 regularization
    # layer_c_xgb_reg_lambda: float = 1.0     # L2 regularization
    # layer_c_xgb_min_child_weight: int = 5   # min samples per leaf
    # layer_c_xgb_gamma: float = 0.1          # min split gain

def make_model():
    """Create XGBoost classifier."""
    params = {"device": "cuda", "tree_method": "hist"} if _DEVICE == "cuda" else {}
    return XGBClassifier(
        random_state=SEED,
        early_stopping_rounds=3000,
        learning_rate=0.05,
        **params
    )