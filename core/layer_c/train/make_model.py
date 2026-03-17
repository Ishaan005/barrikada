import torch
from core.settings import Settings
from xgboost import XGBClassifier

_settings = Settings()
SEED = _settings.layer_c_seed
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_model(scale_pos_weight):
    """Create Layer C XGBoost model with explicit hyperparameters."""
    params = {"device": "cuda", "tree_method": _settings.layer_c_xgb_tree_method} if _DEVICE == "cuda" else {
        "tree_method": _settings.layer_c_xgb_tree_method
    }
    return XGBClassifier(
        random_state=SEED,
        n_estimators=_settings.layer_c_xgb_n_estimators,
        max_depth=_settings.layer_c_xgb_max_depth,
        learning_rate=_settings.layer_c_xgb_learning_rate,
        subsample=_settings.layer_c_xgb_subsample,
        colsample_bytree=_settings.layer_c_xgb_colsample_bytree,
        min_child_weight=_settings.layer_c_xgb_min_child_weight,
        gamma=_settings.layer_c_xgb_gamma,
        reg_alpha=_settings.layer_c_xgb_reg_alpha,
        reg_lambda=_settings.layer_c_xgb_reg_lambda,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=_settings.layer_c_xgb_early_stopping_rounds,
        eval_metric="logloss",
        **params
    )