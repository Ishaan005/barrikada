from core.settings import Settings
from scripts.bundling.assemble_fast_bundle import DIRECTORY_MAPPINGS, FILE_MAPPINGS


def test_fast_bundle_excludes_training_and_pytorch_artifacts():
    destinations = set(FILE_MAPPINGS.values()) | set(DIRECTORY_MAPPINGS.values())
    forbidden = ("classifier.joblib", "model.safetensors", "training_args.bin")
    assert all(not any(name in destination for name in forbidden) for destination in destinations)


def test_fast_bundle_contains_every_runtime_backend():
    destinations = set(FILE_MAPPINGS.values()) | set(DIRECTORY_MAPPINGS.values())
    assert "layer_b/embeddings/prompt_encoder_onnx" in destinations
    assert "layer_c/classifier.onnx" in destinations
    assert "layer_c/encoder_onnx" in destinations
    assert "layer_d/onnx" in destinations


def test_settings_accept_an_onnx_only_fast_bundle(tmp_path, monkeypatch):
    (tmp_path / "layer_c").mkdir()
    (tmp_path / "layer_c" / "classifier.onnx").write_bytes(b"onnx")
    (tmp_path / "layer_d" / "onnx").mkdir(parents=True)
    monkeypatch.setenv("BARRIKADE_CORE_MODELS_DIR", str(tmp_path))

    settings = Settings()
    assert settings.model_path == str(tmp_path / "layer_c" / "classifier.onnx")
    assert settings.layer_d_output_dir == str(tmp_path / "layer_d" / "onnx")
