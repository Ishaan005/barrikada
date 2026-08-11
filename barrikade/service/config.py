"""Environment-backed service settings with legacy prefix compatibility."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path


_WARNED_ALIASES: set[str] = set()


def env_value(name: str, default: str | None = None) -> str | None:
    """Read a canonical setting, falling back to the old misspelled prefix."""
    value = os.getenv(name)
    if value is not None:
        return value

    legacy_name = name.replace("BARRIKADE_", "BARRIKADA_", 1)
    legacy_value = os.getenv(legacy_name)
    if legacy_value is not None:
        if legacy_name not in _WARNED_ALIASES:
            warnings.warn(
                f"{legacy_name} is deprecated; use {name}. Support will be removed "
                "after two minor releases.",
                DeprecationWarning,
                stacklevel=2,
            )
            _WARNED_ALIASES.add(legacy_name)
        return legacy_value
    return default


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServiceSettings:
    database_url: str
    token_file: Path | None
    local_token: str | None
    model_bundle_version: str
    active_profile: str
    inference_workers: int
    inference_queue_size: int
    shutdown_grace_seconds: float
    auto_migrate: bool
    require_artifact_verification: bool
    artifact_manifest: Path | None
    artifact_public_key: Path | None

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        token_file = env_value("BARRIKADE_SERVICE_TOKEN_FILE")
        manifest = env_value("BARRIKADE_BUNDLE_MANIFEST_PATH")
        public_key = env_value("BARRIKADE_BUNDLE_PUBLIC_KEY_PATH")
        return cls(
            database_url=env_value("BARRIKADE_DATABASE_URL", "sqlite:////tmp/barrikade-metadata.db")
            or "sqlite:////tmp/barrikade-metadata.db",
            token_file=Path(token_file).expanduser().resolve() if token_file else None,
            local_token=env_value("BARRIKADE_SERVICE_TOKEN"),
            model_bundle_version=env_value("BARRIKADE_MODEL_BUNDLE_VERSION", "development")
            or "development",
            active_profile=env_value("BARRIKADE_ACTIVE_PROFILE", "jentic_gateway_fast")
            or "jentic_gateway_fast",
            inference_workers=max(1, int(env_value("BARRIKADE_INFERENCE_WORKERS", "1") or 1)),
            inference_queue_size=max(
                0, int(env_value("BARRIKADE_INFERENCE_QUEUE_SIZE", "32") or 32)
            ),
            shutdown_grace_seconds=max(
                0.0, float(env_value("BARRIKADE_SHUTDOWN_GRACE_SECONDS", "15") or 15)
            ),
            auto_migrate=_as_bool(env_value("BARRIKADE_AUTO_MIGRATE"), True),
            require_artifact_verification=_as_bool(env_value("BARRIKADE_VERIFY_ARTIFACTS"), False),
            artifact_manifest=Path(manifest).expanduser().resolve() if manifest else None,
            artifact_public_key=Path(public_key).expanduser().resolve() if public_key else None,
        )
