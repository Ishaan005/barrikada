from pathlib import Path

from core.layer_b import export_layer_b_onnx
from core.layer_c import export_layer_c_encoder_onnx, export_layer_c_onnx
from core.layer_d import export_layer_d_onnx, quantize_layer_d_onnx


ROOT = Path(__file__).resolve().parents[3]


def test_model_exporters_resolve_the_repository_root():
    exporters = (
        export_layer_b_onnx,
        export_layer_c_encoder_onnx,
        export_layer_c_onnx,
        export_layer_d_onnx,
        quantize_layer_d_onnx,
    )
    assert all(module.PROJECT_ROOT == ROOT for module in exporters)
