from pydantic import BaseModel
from pathlib import Path


class Settings(BaseModel):
    app_name: str = "Barrikada"
    debug_mode: bool = False

    # Use absolute paths based on project root
    _project_root = Path(__file__).parent.parent

    ### Layer B (signatures)
    # Scoring/thresholds
    layer_b_allow_confidence: float = 0.99
    layer_b_block_min_hits: int = 4
    layer_b_block_min_rule_precision: float = 1
    layer_b_flag_confidence: float = 0.5
    layer_b_block_confidence: float = 0.95

    # Secondary block: corroborating $s evidence lowers the $re threshold
    layer_b_block_secondary_re_hits: int = 3     # min qualifying $re for secondary block
    layer_b_block_secondary_s_hits: int = 7      # min unique $s hits to corroborate

    # Verdict for inputs with zero malicious matches and not allowlisted
    layer_b_no_match_verdict: str = "flag"        # "allow" or "flag"
    layer_b_no_match_confidence: float = 0.3

    # Allow verdict for very weak evidence (only $s hits, low confidence)
    layer_b_allow_max_s_confidence: float = 0.0   # 0 = disabled; set >0 to enable

    @property
    def layer_b_signatures_extracted_dir(self):
        return str(self._project_root / "core/layer_b" / "signatures" / "extracted")

    @property
    def layer_b_malicious_rules_path(self):
        return str(Path(self.layer_b_signatures_extracted_dir) / "malicious_block_high_signatures.yar")

    @property
    def layer_b_allow_rules_path(self):
        return str(Path(self.layer_b_signatures_extracted_dir) / "safe_allow_signatures.yar")
    

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

