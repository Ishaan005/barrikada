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
    layer_b_block_min_hits: int = 2
    # To make hard-blocks rare/high-precision, require the matched rules to have
    # very high extracted precision (from YARA meta). Otherwise, we only FLAG.
    layer_b_block_min_rule_precision: float = 0.99
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

