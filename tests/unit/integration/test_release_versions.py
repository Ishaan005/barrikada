from pathlib import Path

import yaml

from core.__version__ import __version__


ROOT = Path(__file__).resolve().parents[3]


def test_service_plugin_and_chart_share_one_release_version():
    plugin = (ROOT / "integrations" / "jentic-one" / "pyproject.toml").read_text()
    chart = yaml.safe_load((ROOT / "deploy" / "jentic" / "Chart.yaml").read_text())
    values = yaml.safe_load((ROOT / "deploy" / "jentic" / "values.yaml").read_text())

    assert f'version = "{__version__}"' in plugin
    assert chart["version"] == __version__
    assert chart["appVersion"] == __version__
    assert values["barrikade"]["image"]["tag"] == __version__
    assert values["jentic"]["barrikadeImage"]["tag"] == __version__

    overlay = (ROOT / "deploy" / "jentic" / "overlay.Dockerfile").read_text()
    release = (ROOT / "Dockerfile.release").read_text()
    compose = (ROOT / "deploy" / "compose.jentic.yaml").read_text()
    assert f"ARG BARRIKADE_VERSION={__version__}" in overlay
    assert f"core-runtime:{__version__}" in release
    assert f"barrikade/jentic-one:{__version__}" in compose
    assert f"barrikade/core:{__version__}" in compose
