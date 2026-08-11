"""Import-time registration only: no I/O, secrets, clients, loops, or threads."""

from jentic_one.registry.ingest import PipelineStageSpec, SpecType, register_pipeline_stage
from jentic_one.shared.config import register_config

from barrikade_jentic.config import BarrikadeConfig
from barrikade_jentic.ingest import BarrikadeSpecStage

CONFIG_KEY = "barrikade"
STAGE_KEY = "barrikade.specification"

register_config(CONFIG_KEY, BarrikadeConfig)
register_pipeline_stage(
    PipelineStageSpec(
        name=STAGE_KEY,
        factory=BarrikadeSpecStage,
        spec_types=frozenset({SpecType.OPENAPI}),
    )
)


__all__ = ["CONFIG_KEY", "STAGE_KEY"]
