"""Internal overlay-image bootstrap; users continue to configure Jentic normally."""

from __future__ import annotations

import sys

import uvicorn
from jentic_one import __main__ as jentic_main
from jentic_one.auth.web.app import install_on_app as install_auth_verifier
from jentic_one.broker.web.app import create_app as create_broker_app
from jentic_one.shared.config import load_config
from jentic_one.shared.context import Context
from jentic_one.shared.logging import configure_logging
from jentic_one.shared.metrics import configure_metrics
from jentic_one.shared.tracing import configure_tracing
from jentic_one.shared.web.app_factory import create_combined_app
from jentic_one.shared.web.container import AppContainer
from jentic_one.wiring import install_broker_registry_resolver

import barrikade_jentic.registration  # noqa: F401
from barrikade_jentic.admin import router as admin_router
from barrikade_jentic.broker import install_broker_factory
from barrikade_jentic.config import get_barrikade_config
from barrikade_jentic.runtime import PluginRuntime, install_runtime_lifecycle, set_runtime


def _enabled_config(config):
    derived = config.model_copy(deep=True)
    derived.broker.resilience.upstream.stream_passthrough_enabled = False
    return derived


def _build_enabled_app(config):
    apps = config.apps
    ctx = Context(config, allowed_dbs=jentic_main._expand_allowed_dbs(apps))
    plugin_config = get_barrikade_config(config)
    relevant = any(surface in apps for surface in {"broker", "registry", "admin", "auth"})
    runtime = PluginRuntime(plugin_config, ctx) if relevant else None
    set_runtime(runtime)

    extra_routers = ()
    if runtime is not None and any(surface in apps for surface in {"admin", "auth"}):
        extra_routers = ((admin_router, "/plugins/barrikade", ("Barrikade",)),)
    installers = ()
    if runtime is not None:
        values = [install_runtime_lifecycle]
        if "broker" in apps:
            values.append(install_broker_factory)
        installers = tuple(values)
    container = AppContainer(
        ctx=ctx,
        broker=None,
        extra_routers=extra_routers,
        extra_installers=installers,
    )

    if apps == ["broker"]:
        app = create_broker_app(ctx, container=container)
        if not hasattr(app.state, "verify_token"):
            install_auth_verifier(app, ctx)
        if ctx.is_db_allowed("registry"):
            install_broker_registry_resolver(app, ctx)
        return app
    if len(apps) > 1:
        return create_combined_app(ctx, apps, container=container)

    app = jentic_main._build_app(ctx, apps)
    if runtime is not None:
        if extra_routers:
            app.include_router(admin_router, prefix="/plugins/barrikade", tags=["Barrikade"])
        install_runtime_lifecycle(app, ctx)
    return app


def create_app():
    config = load_config()
    plugin_config = get_barrikade_config(config)
    if not plugin_config.enabled:
        set_runtime(None)
        return jentic_main.create_app()

    config = _enabled_config(config)
    configure_logging(config)
    configure_tracing(jentic_main._service_name(), config.observability.tracing)
    configure_metrics(jentic_main._service_name(), config.observability.metrics)
    return _build_enabled_app(config)


def serve() -> None:
    config = load_config()
    plugin_config = get_barrikade_config(config)
    if not plugin_config.enabled:
        set_runtime(None)
        jentic_main._serve()
        return

    config = _enabled_config(config)
    configure_logging(config)
    configure_tracing(jentic_main._service_name(), config.observability.tracing)
    configure_metrics(jentic_main._service_name(), config.observability.metrics)
    app = _build_enabled_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] not in {"serve"}:
        return jentic_main.main(arguments)
    serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
