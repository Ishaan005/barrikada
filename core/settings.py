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
    layer_b_block_threshold: float = 0.78   # attack sim above this → BLOCK (with contrastive guard)
    layer_b_flag_threshold: float = 0.55    # attack sim above this → FLAG
    # Below flag_threshold → SAFE (allow)

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

    @property
    def layer_b_signatures_dir(self):
        return str(self._project_root / "core" / "layer_b" / "signatures" / "embeddings")
    

    ### Layer C
    # Routing thresholds for Layer C classifier:
    layer_c_low_threshold: float = 0.25
    layer_c_high_threshold: float = 0.75

    layer_c_seed: int = 42

    layer_c_val_test_size: float = 0.30
    layer_c_test_split: float = 0.50

    layer_c_embedding_model: str = "all-mpnet-base-v2"
    layer_c_embedding_batch_size: int    = 128

    # Probability calibration
    layer_c_calibration_method: str = "isotonic"

    # Threshold tuning constraints
    layer_c_tune_target_block_precision: float = 0.90
    layer_c_tune_max_malicious_allow_rate: float = 0.05  # hard cap: malicious texts reaching allow
    layer_c_tune_max_safe_fpr: float = 0.10          # hard cap: safe texts flagged or blocked
    layer_c_tune_min_flag_band: float = 0.05

    # Threshold grid search
    layer_c_tune_low_grid_start: float = 0.10
    layer_c_tune_low_grid_end: float = 0.50
    layer_c_tune_high_grid_start: float = 0.50
    layer_c_tune_high_grid_end: float = 0.99
    layer_c_tune_grid_steps: int = 200

    # Composite score weights — objective: maximise block_recall, penalise FN (mal→allow) and FP (safe→flag/block)
    layer_c_tune_w_block_recall: float = 0.5       # reward: malicious texts that are blocked
    layer_c_tune_w_mal_allow_penalty: float = 0.4  # penalty: malicious texts reaching allow (false negatives)
    layer_c_tune_w_safe_fpr_penalty: float = 0.3   # penalty: safe texts flagged or blocked (false positives)

    @property
    def dataset_path(self):
        return str(self._project_root / "datasets" / "barrikada.csv")
    
    @property
    def model_path(self):
        return str(self._project_root / "core/layer_c" / "outputs" / "classifier.joblib")

