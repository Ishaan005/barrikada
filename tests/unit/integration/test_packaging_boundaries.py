import ast
import re
import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = ROOT / "integrations" / "jentic-one"


def _dependency_name(requirement: str) -> str:
    return re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].strip().lower()


def test_thin_plugin_has_only_approved_runtime_dependencies():
    project = tomllib.loads((PLUGIN_ROOT / "pyproject.toml").read_text())["project"]
    assert {_dependency_name(value) for value in project["dependencies"]} == {
        "httpx",
        "jentic-one",
        "pydantic",
    }


def test_thin_plugin_never_imports_core_or_model_runtimes():
    forbidden = {"barrikade", "core", "faiss", "onnxruntime", "torch", "transformers", "xgboost"}
    imported = set()
    for source_file in (PLUGIN_ROOT / "src" / "barrikade_jentic").rglob("*.py"):
        tree = ast.parse(source_file.read_text(), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
    assert imported.isdisjoint(forbidden)


def test_overlay_is_pinned_and_contains_only_the_plugin_wheel():
    dockerfile = (ROOT / "deploy" / "jentic" / "overlay.Dockerfile").read_text()
    assert "jentic-one-app@sha256:" in dockerfile
    assert dockerfile.index("ARG JENTIC_BASE=") < dockerfile.index("FROM python:")
    assert "pip install --no-cache-dir --no-deps" in dockerfile
    assert "COPY core" not in dockerfile
    assert "COPY barrikade " not in dockerfile


def test_compose_exposes_one_public_barrikade_switch():
    compose = yaml.safe_load((ROOT / "deploy" / "compose.jentic.yaml").read_text())
    for service_name in ("jentic-app", "jentic-broker"):
        environment = compose["services"][service_name]["environment"]
        assert environment["JENTIC__BARRIKADE__ENABLED"] == ("${JENTIC__BARRIKADE__ENABLED:-false}")
    assert "BARRIKADE_JENTIC_IMAGE" not in str(compose["services"]["jentic-app"]["environment"])


def test_release_image_requires_external_signed_bundle_contexts():
    dockerfile = (ROOT / "Dockerfile.release").read_text()
    assert "COPY --from=bundle" in dockerfile
    assert "COPY --from=public-key" in dockerfile
    assert "USER barrikade" in dockerfile
