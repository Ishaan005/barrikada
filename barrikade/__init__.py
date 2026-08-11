"""Public SDK for Barrikade.

Importing this package is intentionally side-effect free. Artifact verification and model
loading happen only when an SDK pipeline or the service application is explicitly started.
"""

from barrikade.sdk import (
    IncidentReport,
    InMemorySessionStore,
    InputProvenance,
    Intervention,
    PIPipeline,
    SessionDetectResult,
    SessionEvent,
    SessionEventType,
    SessionNotActiveError,
    SessionOrchestrator,
    SessionSettings,
    SessionStatus,
    SessionStoreBackend,
    WorkloadSession,
    create_session_orchestrator,
)
from core.__version__ import __version__
from core.artifacts import (
    ArtifactDownloadError,
    download_runtime_artifacts,
    download_runtime_bundle,
    ensure_runtime_artifacts,
    ensure_runtime_bundle,
)


__all__ = [
    "__version__",
    "ArtifactDownloadError",
    "PIPipeline",
    "download_runtime_bundle",
    "download_runtime_artifacts",
    "ensure_runtime_bundle",
    "ensure_runtime_artifacts",
    "SessionOrchestrator",
    "create_session_orchestrator",
    "SessionDetectResult",
    "SessionSettings",
    "SessionEvent",
    "SessionEventType",
    "SessionNotActiveError",
    "SessionStatus",
    "WorkloadSession",
    "SessionStoreBackend",
    "InMemorySessionStore",
    "InputProvenance",
    "Intervention",
    "IncidentReport",
]
