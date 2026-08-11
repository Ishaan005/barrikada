"""Hashed bearer-token authentication for service-to-service API calls."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path


class AuthenticationError(RuntimeError):
    pass


class AuthorizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Principal:
    id: str
    scopes: frozenset[str]

    @property
    def storage_hash(self) -> str:
        return hashlib.sha256(self.id.encode()).hexdigest()


class TokenVerifier:
    def __init__(self, entries: dict[str, Principal] | None = None) -> None:
        self._entries = entries or {}

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    @classmethod
    def from_file(cls, path: Path) -> "TokenVerifier":
        document = json.loads(path.read_text(encoding="utf-8"))
        entries: dict[str, Principal] = {}
        for item in document.get("tokens", []):
            token = item.get("token")
            principal_id = item.get("principal_id")
            scopes = item.get("scopes")
            if not isinstance(token, str) or not token:
                raise ValueError("token file contains an invalid token")
            if not isinstance(principal_id, str) or not principal_id:
                raise ValueError("token file contains an invalid principal_id")
            if not isinstance(scopes, list) or not all(isinstance(scope, str) for scope in scopes):
                raise ValueError("token file contains invalid scopes")
            entries[cls._digest(token)] = Principal(principal_id, frozenset(scopes))
        if not entries:
            raise ValueError("token file contains no tokens")
        return cls(entries)

    @classmethod
    def local(cls, token: str, principal_id: str = "jentic-local") -> "TokenVerifier":
        scopes = frozenset({"assessments:write", "assessments:read", "overrides:write"})
        return cls({cls._digest(token): Principal(principal_id, scopes)})

    def authenticate(self, authorization: str | None, required_scope: str) -> Principal:
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthenticationError("Bearer authentication is required")
        token = authorization[7:]
        if not token:
            raise AuthenticationError("Bearer authentication is required")

        candidate = self._digest(token)
        principal = None
        for expected, entry in self._entries.items():
            if hmac.compare_digest(candidate, expected):
                principal = entry
        if principal is None:
            raise AuthenticationError("The service token is invalid")
        if required_scope not in principal.scopes:
            raise AuthorizationError("The service token lacks the required scope")
        return principal
