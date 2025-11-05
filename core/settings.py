from pydantic import BaseModel
from pathlib import Path


class Settings(BaseModel):
    app_name: str = "Barrikada"
    debug_mode: bool = False
    
    # Use absolute paths based on project root
    _project_root: Path = Path(__file__).parent.parent
    
    @property
    def dataset_path(self) -> str:
        return str(self._project_root / "datasets" / "barrikada.csv")
    
    @property
    def model_path(self) -> str:
        return str(self._project_root / "core/layer_c" / "models" / "tf_idf_logreg.joblib")
    
    @property
    def vectorizer_path(self) -> str:
        return str(self._project_root / "core/layer_c" / "models" / "tf_idf_vectorizer.joblib")