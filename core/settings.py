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
    @property
    def dataset_path(self) -> str:
        return str(self._project_root / "datasets" / "barrikada.csv")
    
    @property
    def model_path(self) -> str:
        return str(self._project_root / "core/layer_c" / "outputs" / "tf_idf_logreg.joblib")
    
    @property
    def vectorizer_path(self) -> str:
        return str(self._project_root / "core/layer_c" / "outputs" / "tf_idf_vectorizer.joblib")

