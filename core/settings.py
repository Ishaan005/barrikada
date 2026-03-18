from pydantic import BaseModel
from pathlib import Path


class Settings(BaseModel):
    app_name: str = "Barrikada"
    debug_mode: bool = False

    # Use absolute paths based on project root
    _project_root = Path(__file__).parent.parent

    ### Layer B (embedding-based contrastive signature engine)
    layer_b_embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Two-threshold decision system (applied to mean top-k attack similarity)
    # Empirically calibrated on barrikada_test.csv (block_prec=0.96, fblk=0.53%)
    layer_b_block_threshold: float = 0.86   # sweep-calibrated conservative block threshold
    layer_b_flag_threshold: float = 0.62    # sweep-calibrated conservative flag threshold
    # Below flag_threshold → SAFE (allow)

    # Contrastive guardrails for safer Layer B calibration
    layer_b_block_min_margin: float = 0.08  # require strong attack dominance for hard block
    layer_b_enable_safe_recovery: bool = True
    layer_b_safe_recovery_max_attack_sim: float = 0.63
    layer_b_safe_recovery_min_benign_sim: float = 0.80
    layer_b_safe_recovery_max_margin: float = -0.10  # allow only when benign clearly dominates

    # Top-k similarity aggregation
    layer_b_top_k: int = 5

    # Cluster building
    layer_b_n_clusters: int = 64
    layer_b_min_cluster_purity: float = 0.70  # drop clusters below this purity
    layer_b_purity_proximity: float = 0.70     # only count benign prompts with sim >= this

    # Confidence values emitted in LayerBResult
    layer_b_block_confidence: float = 0.95
    layer_b_flag_confidence: float = 0.50
    layer_b_safe_confidence: float = 0.10

    # Dual-encoder contrastive training hyperparameters
    layer_b_dual_encoder_temperature: float = 0.05
    layer_b_dual_encoder_epochs: int = 3
    layer_b_dual_encoder_batch_size: int = 8       # small batch, use grad_accum to compensate
    layer_b_dual_encoder_lr: float = 2e-5
    layer_b_dual_encoder_hard_negatives: int = 3
    layer_b_dual_encoder_max_samples: int = 50000
    layer_b_dual_encoder_grad_accum_steps: int = 8  # effective batch = 8 * 8 = 64
    layer_b_dual_encoder_use_amp: bool = True       # mixed precision for memory savings

    @property
    def layer_b_signatures_dir(self):
        return str(self._project_root / "core" / "layer_b" / "signatures" / "embeddings")
    

    ### Layer C
    # Routing thresholds for Layer C classifier:
    layer_c_low_threshold: float = 0.05
    layer_c_high_threshold: float = 0.95

    layer_c_seed: int = 42

    layer_c_val_test_size: float = 0.30
    layer_c_test_split: float = 0.50

    layer_c_embedding_model: str = "all-mpnet-base-v2"
    layer_c_embedding_batch_size: int    = 128

    # Probability calibration (fixed to isotonic in training flow)
    layer_c_calibration_bins: int = 15

    # XGBoost configuration for Layer C
    layer_c_xgb_n_estimators: int = 3000
    layer_c_xgb_max_depth: int = 7
    layer_c_xgb_learning_rate: float = 0.05
    layer_c_xgb_subsample: float = 0.8
    layer_c_xgb_colsample_bytree: float = 0.9
    layer_c_xgb_scale_pos_multiplier: float = 1.5
    layer_c_xgb_early_stopping_rounds: int = 150
    layer_c_xgb_tree_method: str = "hist"
    layer_c_xgb_reg_alpha: float = 0.1
    layer_c_xgb_reg_lambda: float = 1.0
    layer_c_xgb_min_child_weight: int = 5
    layer_c_xgb_gamma: float = 0.1

    # Layer C hard-negative mining (train-split SAFE examples near/inside uncertain band)
    layer_c_hard_negative_use_routing_band: bool = True
    layer_c_hard_negative_score_min: float = 0.20
    layer_c_hard_negative_score_max: float = 0.80
    layer_c_hard_negative_max_samples: int = 5000
    layer_c_hard_negative_min_samples: int = 32
    layer_c_hard_negative_augment_multiplier: int = 1

    @property
    def dataset_path(self):
        return str(self._project_root / "datasets" / "barrikada.csv")
    
    @property
    def model_path(self):
        return str(self._project_root / "core/layer_c" / "outputs" / "classifier.joblib")

    ### Layer D (ModernBERT classifier)
    layer_d_model_id: str = "answerdotai/ModernBERT-large"
    layer_d_max_length: int = 512
    layer_d_num_train_epochs: int = 3
    layer_d_learning_rate: float = 2e-5
    layer_d_warmup_ratio: float = 0.06
    layer_d_weight_decay: float = 0.01
    layer_d_malicious_class_weight: float = 1.75
    layer_d_focal_gamma: float = 1.5
    layer_d_train_batch_size: int = 8
    layer_d_eval_batch_size: int = 16
    layer_d_gradient_accumulation_steps: int = 4
    layer_d_gradient_checkpointing: bool = True
    layer_d_dataloader_num_workers: int = 4
    layer_d_use_bf16: bool = True
    layer_d_use_tf32: bool = True
    layer_d_split_train: float = 0.80
    layer_d_split_val: float = 0.10
    layer_d_split_test: float = 0.10
    layer_d_low_threshold: float = 0.20
    layer_d_high_threshold: float = 0.80

    @property
    def layer_d_output_dir(self):
        return str(self._project_root / "core" / "layer_d" / "outputs" / "model")

    @property
    def layer_d_report_path(self):
        return str(self._project_root / "test_results" / "layer_d_eval_latest.json")

