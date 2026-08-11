import asyncio
from pathlib import Path

from jentic_one import __main__ as jentic_main
from jentic_one.shared.config import load_config

from barrikade_jentic import bootstrap
from barrikade_jentic.runtime import get_runtime, set_runtime

JENTIC_CONFIG = (
    Path(__file__).resolve().parents[6]
    / "Barrikade"
    / "Repos"
    / "jentic-one"
    / "config"
    / "local-sqlite.yaml"
)


def _route_shapes(app):
    return sorted(
        (getattr(route, "path", ""), tuple(sorted(getattr(route, "methods", ()) or ())))
        for route in app.routes
    )


def test_disabled_overlay_uses_stock_construction(monkeypatch):
    monkeypatch.setenv("JENTIC_CONFIG_FILE", str(JENTIC_CONFIG))
    monkeypatch.setenv("JENTIC__BARRIKADE__ENABLED", "false")

    plugin_app = bootstrap.create_app()
    stock_app = jentic_main.create_app()

    assert _route_shapes(plugin_app) == _route_shapes(stock_app)
    assert get_runtime() is None
    assert not hasattr(plugin_app.state, "broker_factory")
    assert load_config().broker.resilience.upstream.stream_passthrough_enabled is True


def test_enabled_bootstrap_adds_admin_routes_and_disables_streaming_in_memory(
    monkeypatch, tmp_path
):
    token_file = tmp_path / "token"
    token_file.write_text("local-token")
    monkeypatch.setenv("JENTIC_CONFIG_FILE", str(JENTIC_CONFIG))
    monkeypatch.setenv("JENTIC__BARRIKADE__ENABLED", "true")
    monkeypatch.setenv("JENTIC__BARRIKADE__TOKEN_FILE", str(token_file))

    app = bootstrap.create_app()
    runtime = get_runtime()

    assert runtime is not None
    assert app.state.ctx.config.broker.resilience.upstream.stream_passthrough_enabled is False
    assert load_config().broker.resilience.upstream.stream_passthrough_enabled is True
    assert "/plugins/barrikade/status" in app.openapi()["paths"]

    asyncio.run(runtime.client.close())
    set_runtime(None)


def test_enabled_standalone_broker_installs_wrapper_factory(monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("local-token")
    monkeypatch.setenv("JENTIC_CONFIG_FILE", str(JENTIC_CONFIG))
    monkeypatch.setenv("JENTIC__APPS", "broker")
    monkeypatch.setenv("JENTIC__BARRIKADE__ENABLED", "true")
    monkeypatch.setenv("JENTIC__BARRIKADE__TOKEN_FILE", str(token_file))

    app = bootstrap.create_app()
    runtime = get_runtime()

    assert runtime is not None
    assert callable(app.state.broker_factory)
    asyncio.run(runtime.client.close())
    set_runtime(None)
