"""Registered Jentic configuration with a one-boolean public surface."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BarrikadeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False

    # Operator-owned defaults. The umbrella chart supplies/protects these when enabled.
    endpoint: str = "http://barrikade-core:8000"
    token_file: Path = Path("/var/run/secrets/barrikade/token")
    request_timeout_seconds: float = Field(default=1.0, gt=0.0, le=30.0)
    max_text_bytes: int = Field(default=2 * 1024 * 1024, ge=1024, le=2 * 1024 * 1024)
    max_segment_bytes: int = Field(default=64 * 1024, ge=1024, le=64 * 1024)
    max_segments: int = Field(default=256, ge=1, le=256)
    event_queue_size: int = Field(default=256, ge=1, le=10_000)
    decompression_ratio_limit: int = Field(default=50, ge=1, le=100)
    runtime_profile: Literal["jentic_gateway_fast"] = "jentic_gateway_fast"
    specification_profile: Literal["jentic_spec"] = "jentic_spec"
    enforcement_policy: Literal["balanced"] = "balanced"


def get_barrikade_config(app_config) -> BarrikadeConfig:
    value = app_config.extension("barrikade") if app_config is not None else None
    return value if isinstance(value, BarrikadeConfig) else BarrikadeConfig()
