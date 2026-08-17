# Assessment API v2

## Responsibility

This document freezes the HTTP and persistence contract shared by Core and the generated
Jentic client.

## Authentication

All `/v2` routes except health require bearer authentication. Assessment tokens carry the
`assessments:write` scope. Override mutation requires `overrides:write`; assessment reads
require `assessments:read`. Missing, invalid, and insufficient credentials return 401 and 403
respectively with RFC 9457 problem responses.

## `POST /v2/assessments`

The request contains:

- `request_id`: stable opaque identifier, 1-128 characters;
- `profile`: closed profile identifier;
- `deadline_ms`: remaining client budget, 1-30,000 ms;
- `content_sha256`: lowercase 64-character digest;
- `segments`: 1-256 structured text segments;
- `context`: optional closed correlation object.

Each segment contains a stable `id`, `text`, closed `source`, optional safe `locator`, and
`sha256`. A segment is at most 64 KiB encoded as UTF-8; aggregate text is at most 2 MiB; a
locator is at most 512 characters. Duplicate segment IDs, digest mismatch, unknown sources,
unknown context properties, and invalid profiles return 422. Limits return 413. Context may
contain upstream status, content type, Jentic execution ID, API ID, operation ID, and surface;
it cannot contain headers, credentials, cookies, free-form values, or actor email.

The successful response contains assessment ID, `complete`/`partial`,
`allow`/`flag`/`block`/`unknown`, calibrated risk score, closed categories, finding-to-segment
references, deciding layer, profile, bundle version, processing time, and optional applied
override metadata. Findings never contain matched text.

Idempotency is keyed by authenticated caller plus `request_id`. The same request ID and
overall digest returns the existing response. Reuse with a different digest, profile, or
canonical segment metadata returns 409. Concurrent duplicates converge on one stored row.

## Errors

| Status | Meaning |
|---|---|
| 401 | missing or invalid authentication |
| 403 | authenticated token lacks required scope |
| 409 | idempotency key reused for different content |
| 413 | segment count, segment size, aggregate size, or locator limit exceeded |
| 422 | malformed request, digest mismatch, invalid profile or closed value |
| 429 | inference admission queue is full |
| 503 | profile/model/persistence unavailable |
| 504 | deadline expired before a complete verdict |

Problems contain `type`, `title`, `status`, a generic `detail`, and safe correlation IDs.
They never echo content, locator values on authentication failures, or exception strings.

## Storage schema

Barrikade owns a logical database and a forward-only migration history. PostgreSQL is the
production engine; SQLite is supported only for local Compose and tests.

Assessment rows retain IDs, caller hash, request/content/segment digests, profile, status,
verdict, score, categories, deciding layer, bundle version, safe locators, timing, and
operational state. They do not retain text, snippets, prompts, completions, headers,
credentials, cookies, or Jentic actor emails. Assessment metadata defaults to 30 days.

Override rows retain assessment ID, exact content digest, profile, bundle version, actor ID,
reason, created/resolved/revoked timestamps, and audit state. Retention follows the security
audit policy.

## Overrides

Core provides authenticated create, get/resolve, and revoke operations. An active override
is usable only when assessment ID, content digest, profile, and model bundle version all
match. Reassessment with a changed bundle invalidates it. Revocation is append-only audit
state, not destructive deletion.

Jentic passes its authenticated administrator's stable actor ID, never email. The Jentic
admin facade performs its own `org:admin` check before calling Core.

## Contract generation

Core's OpenAPI document is the source of truth. CI exports a canonical document, regenerates
the client under `integrations/jentic-one/src/barrikade_jentic/generated`, and fails on any
diff. The client is intentionally limited to `httpx` and Pydantic and carries an explicit API
contract version header.

## Completion criteria

- Schema, limit, authentication, idempotency, overload, and deadline tests cover every status.
- Concurrent idempotency tests return one assessment.
- Database and telemetry scans find no raw request text.
- Override tests cover exact match, changed digest/profile/bundle, resolution, and revocation.
- Generated-client regeneration is deterministic and clean in CI.
