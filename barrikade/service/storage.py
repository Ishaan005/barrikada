"""Metadata-only persistence for assessments and exact-bound overrides."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from barrikade.service.schemas import (
    AssessmentProfile,
    AssessmentRequest,
    AssessmentResponse,
    OverrideCreateRequest,
    OverrideResponse,
)


class StorageUnavailableError(RuntimeError):
    pass


class IdempotencyConflictError(RuntimeError):
    pass


class MetadataStore:
    """Small DB-API store supporting SQLite locally and PostgreSQL in production."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._dialect, self._target = self._parse_url(database_url)

    @staticmethod
    def _parse_url(database_url: str) -> tuple[str, str]:
        if database_url.startswith("sqlite:///"):
            target = database_url.removeprefix("sqlite:///")
            if target == ":memory:":
                return "sqlite", target
            return "sqlite", str(Path(target).resolve())
        if database_url.startswith(("postgresql://", "postgres://")):
            return "postgresql", database_url
        raise ValueError("BARRIKADE_DATABASE_URL must use sqlite:/// or postgresql://")

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self._dialect == "sqlite":
            connection = sqlite3.connect(self._target, timeout=10)
            connection.row_factory = sqlite3.Row
        else:
            try:
                import psycopg  # noqa: PLC0415
                from psycopg.rows import dict_row  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - depends on production extra
                raise StorageUnavailableError(
                    "PostgreSQL support requires the barrikade service dependency group"
                ) from exc
            connection = psycopg.connect(self._target, row_factory=dict_row)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _sql(self, statement: str) -> str:
        return statement if self._dialect == "sqlite" else statement.replace("?", "%s")

    def migrate(self) -> None:
        if self._dialect == "sqlite" and self._target != ":memory:":
            Path(self._target).parent.mkdir(parents=True, exist_ok=True)
        statements = (
            """
            CREATE TABLE IF NOT EXISTS barrikade_schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS barrikade_assessments (
                assessment_id TEXT PRIMARY KEY,
                caller_hash TEXT NOT NULL,
                request_id TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                request_fingerprint TEXT NOT NULL,
                profile TEXT NOT NULL,
                response_json TEXT NOT NULL,
                segment_metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                UNIQUE(caller_hash, request_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS barrikade_overrides (
                override_id TEXT PRIMARY KEY,
                assessment_id TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                profile TEXT NOT NULL,
                model_bundle_version TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                revoked_at TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS barrikade_override_binding_idx
            ON barrikade_overrides (
                assessment_id, content_sha256, profile, model_bundle_version, state
            )
            """,
        )
        with self._connect() as connection:
            cursor = connection.cursor()
            for statement in statements:
                cursor.execute(statement)
            cursor.execute(
                self._sql(
                    "INSERT INTO barrikade_schema_migrations(version, applied_at) "
                    "SELECT 1, ? WHERE NOT EXISTS "
                    "(SELECT 1 FROM barrikade_schema_migrations WHERE version = 1)"
                ),
                (datetime.now(UTC).isoformat(),),
            )

    @staticmethod
    def _row_value(row: Any, key: str) -> Any:
        return row[key]

    def get_assessment_by_request(
        self, caller_hash: str, request_id: str
    ) -> tuple[str, str, AssessmentResponse] | None:
        with self._connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT content_sha256, request_fingerprint, response_json "
                    "FROM barrikade_assessments WHERE caller_hash = ? AND request_id = ?"
                ),
                (caller_hash, request_id),
            ).fetchone()
        if row is None:
            return None
        response = AssessmentResponse.model_validate_json(self._row_value(row, "response_json"))
        response.override = self.get_active_override(
            response.assessment_id,
            self._row_value(row, "content_sha256"),
            response.profile,
            response.model_bundle_version,
        )
        return (
            self._row_value(row, "content_sha256"),
            self._row_value(row, "request_fingerprint"),
            response,
        )

    def get_assessment(self, assessment_id: str) -> AssessmentResponse | None:
        with self._connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT content_sha256, response_json FROM barrikade_assessments "
                    "WHERE assessment_id = ?"
                ),
                (assessment_id,),
            ).fetchone()
        if row is None:
            return None
        response = AssessmentResponse.model_validate_json(self._row_value(row, "response_json"))
        response.override = self.get_active_override(
            response.assessment_id,
            self._row_value(row, "content_sha256"),
            response.profile,
            response.model_bundle_version,
        )
        return response

    def save_assessment(
        self,
        caller_hash: str,
        request: AssessmentRequest,
        response: AssessmentResponse,
        retention_days: int = 30,
    ) -> AssessmentResponse:
        created_at = datetime.now(UTC)
        segment_metadata = [
            {
                "id": segment.id,
                "sha256": segment.sha256,
                "source": segment.source.value,
                "locator": segment.locator,
            }
            for segment in request.segments
        ]
        params = (
            response.assessment_id,
            caller_hash,
            request.request_id,
            request.content_sha256,
            request.idempotency_fingerprint(),
            request.profile.value,
            response.model_dump_json(),
            json.dumps(segment_metadata, sort_keys=True, separators=(",", ":")),
            created_at.isoformat(),
            (created_at + timedelta(days=retention_days)).isoformat(),
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    self._sql(
                        "INSERT INTO barrikade_assessments(assessment_id, caller_hash, "
                        "request_id, content_sha256, request_fingerprint, profile, response_json, "
                        "segment_metadata_json, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, "
                        "?, ?, ?, ?)"
                    ),
                    params,
                )
        except Exception as exc:
            existing = self.get_assessment_by_request(caller_hash, request.request_id)
            if existing is None:
                raise StorageUnavailableError("assessment metadata could not be stored") from exc
            digest, fingerprint, saved = existing
            if digest != request.content_sha256 or fingerprint != request.idempotency_fingerprint():
                raise IdempotencyConflictError("request ID was already used") from exc
            return saved
        return response

    def create_override(self, request: OverrideCreateRequest) -> OverrideResponse:
        binding = self.get_assessment_binding(request.assessment_id)
        if binding is None:
            raise KeyError(request.assessment_id)
        content_sha256, assessment = binding
        if (
            content_sha256 != request.content_sha256
            or assessment.profile != request.profile
            or assessment.model_bundle_version != request.model_bundle_version
        ):
            raise IdempotencyConflictError("override does not match the assessment binding")
        created_at = datetime.now(UTC).isoformat()
        override = OverrideResponse(
            id=f"ovr_{uuid4().hex}",
            assessment_id=request.assessment_id,
            content_sha256=request.content_sha256,
            profile=request.profile,
            model_bundle_version=request.model_bundle_version,
            actor_id=request.actor_id,
            reason=request.reason,
            state="active",
            created_at=created_at,
        )
        with self._connect() as connection:
            connection.execute(
                self._sql(
                    "INSERT INTO barrikade_overrides(override_id, assessment_id, "
                    "content_sha256, profile, model_bundle_version, actor_id, reason, state, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    override.id,
                    override.assessment_id,
                    override.content_sha256,
                    override.profile.value,
                    override.model_bundle_version,
                    override.actor_id,
                    override.reason,
                    override.state,
                    override.created_at,
                ),
            )
        return override

    def get_assessment_binding(self, assessment_id: str) -> tuple[str, AssessmentResponse] | None:
        with self._connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT content_sha256, response_json FROM barrikade_assessments "
                    "WHERE assessment_id = ?"
                ),
                (assessment_id,),
            ).fetchone()
        if row is None:
            return None
        return (
            self._row_value(row, "content_sha256"),
            AssessmentResponse.model_validate_json(self._row_value(row, "response_json")),
        )

    def get_active_override(
        self,
        assessment_id: str,
        content_sha256: str,
        profile: AssessmentProfile,
        model_bundle_version: str,
    ) -> Any | None:
        with self._connect() as connection:
            row = connection.execute(
                self._sql(
                    "SELECT * FROM barrikade_overrides WHERE assessment_id = ? "
                    "AND content_sha256 = ? AND profile = ? AND model_bundle_version = ? "
                    "AND state = 'active' ORDER BY created_at DESC"
                ),
                (assessment_id, content_sha256, profile.value, model_bundle_version),
            ).fetchone()
        if row is None:
            return None
        from barrikade.service.schemas import AppliedOverride  # noqa: PLC0415

        return AppliedOverride(
            id=self._row_value(row, "override_id"),
            actor_id=self._row_value(row, "actor_id"),
            reason=self._row_value(row, "reason"),
            created_at=self._row_value(row, "created_at"),
        )

    def transition_override(self, override_id: str, state: str) -> OverrideResponse:
        if state not in {"resolved", "revoked"}:
            raise ValueError("invalid override transition")
        timestamp_column = "resolved_at" if state == "resolved" else "revoked_at"
        timestamp = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                self._sql(
                    f"UPDATE barrikade_overrides SET state = ?, {timestamp_column} = ? "
                    "WHERE override_id = ? AND state = 'active'"
                ),
                (state, timestamp, override_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(override_id)
            row = connection.execute(
                self._sql("SELECT * FROM barrikade_overrides WHERE override_id = ?"),
                (override_id,),
            ).fetchone()
        return OverrideResponse(
            id=self._row_value(row, "override_id"),
            assessment_id=self._row_value(row, "assessment_id"),
            content_sha256=self._row_value(row, "content_sha256"),
            profile=AssessmentProfile(self._row_value(row, "profile")),
            model_bundle_version=self._row_value(row, "model_bundle_version"),
            actor_id=self._row_value(row, "actor_id"),
            reason=self._row_value(row, "reason"),
            state=self._row_value(row, "state"),
            created_at=self._row_value(row, "created_at"),
            resolved_at=self._row_value(row, "resolved_at"),
            revoked_at=self._row_value(row, "revoked_at"),
        )

    def purge_expired_assessments(self, now: datetime | None = None) -> int:
        cutoff = (now or datetime.now(UTC)).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                self._sql("DELETE FROM barrikade_assessments WHERE expires_at < ?"),
                (cutoff,),
            )
            return cursor.rowcount
