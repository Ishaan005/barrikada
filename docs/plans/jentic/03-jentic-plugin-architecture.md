# Jentic plugin architecture

## Responsibility

This document defines how the thin plugin composes with Jentic One 0.29.1 without modifying
Jentic files, wheels, configuration, migrations, transports, or persistence.

## Thin-package boundary

`barrikade-jentic` depends only on `jentic-one>=0.29.1,<0.30`, `httpx`, and `pydantic`.
It contains configuration, registration, bootstrap, extraction, policy, Core client, event
queue, admin routes, and Jentic adapters. It cannot import the root Barrikade package or any
model/training runtime.

## Registration autoload

The overlay installs a minimal interpreter-startup bootstrap that imports
`barrikade_jentic.registration`. Registration is deliberately inert: it registers the
`barrikade` config model and the Registry ingest stage and defines constants only. It performs
no network or file I/O, reads no secrets, creates no event loop/client/thread, and cannot fail
because protection is disabled.

This registration occurs before every Jentic `load_config()`, including the unchanged Jentic
migration command. Unknown `barrikade` configuration therefore never breaks migrations.

## Application bootstrap and process matrix

The overlay image's internal command invokes the plugin bootstrap. It loads registered
configuration, checks `barrikade.enabled`, and delegates directly to stock Jentic construction
when false. When true, it deep-copies configuration, disables response streaming passthrough,
constructs Jentic's normal `Context`, and composes through the existing factories and
`AppContainer`.

| Jentic process | Plugin components when enabled |
|---|---|
| combined app | lifecycle, admin/status routers, Registry ingest stage |
| Broker | Broker factory installer, scanner lifecycle, event queue |
| split Registry | registered ingest stage, scanner lifecycle |
| split Admin/Auth | status, assessment and override routers |
| split Control | configuration registration only |
| migration Job | configuration registration only |

The installer sets Jentic's supported application-state Broker factory so both sync and async
execution paths create `BarrikadeBroker(DefaultBroker(jentic_composed_runner))`. Jentic remains
responsible for its HTTP pool, credential/SigV4 handling, deadlines, retries, circuit breaker,
rate limits, and workers.

## Buffered Broker flow

1. Copy outbound headers and replace `Accept-Encoding` with `identity`.
2. Delegate execution to Jentic's normal Broker.
3. Classify status, content type, headers, and body without collecting request credentials.
4. If necessary, produce a bounded decompressed scan copy.
5. Extract bounded text segments and submit one assessment.
6. Queue a metadata-only local Jentic event.
7. Pass the original/normalized result or raise an existing exact Jentic error type.

If an upstream ignores identity encoding, gzip/deflate decompression is bounded by compressed
size, expanded size, and expansion ratio. A decoded response removes stale content-encoding
and content-length headers. Unsupported/unsafe compression of otherwise eligible content is
incomplete and fails closed.

Eligible content includes JSON and `application/*+json`, all `text/*`, HTML, XML, YAML, RFC
problem responses, and every HTTP status. JSON keys and string leaves become segments with
JSON-path locators. Invalid JSON falls back to bounded raw-text windows. Long strings use
overlapping windows. Empty and binary content is not applicable. More than two MiB of eligible
text is incomplete.

## Runtime policy and errors

| Assessment | Behavior |
|---|---|
| allow | pass unchanged |
| flag | pass and queue `barrikade.content_flagged` |
| block | raise Jentic `ActionDeniedError` (403) |
| incomplete eligible content | raise Jentic `ActionDeniedError` (403) |
| scanner unavailable/deadline | raise Jentic `RunnerUnavailableError` (503) |
| binary/not applicable | pass unchanged |

The plugin instantiates, and never subclasses, Jentic's existing errors because Jentic maps
exact exception classes. Block metadata contains only a generic type/detail, assessment ID,
categories, bundle version, upstream status, and fatal directive. It never includes matched
content. A defensive `execute_streaming` fails closed if reached while enabled; when disabled
the wrapper is not installed and stock streaming remains active.

## Registry ingest stage

The registered stage returns immediately unless enabled. It canonicalizes the specification,
computes its digest, and extracts agent-visible strings from API/operation/schema/server/
security/callback/webhook metadata with safe field locators. It sends one `jentic_spec`
assessment. Allow or an exact active override continues. Flag, block, partial, oversized, or
scanner failure raises `IngestStageError`, allowing Jentic to roll back the transaction.
Errors expose only assessment ID and safe locators.

## Administration and events

Combined and Admin/Auth processes mount:

- `GET /plugins/barrikade/status`
- `GET /plugins/barrikade/assessments/{id}`
- `POST /plugins/barrikade/spec-overrides`
- `DELETE /plugins/barrikade/spec-overrides/{id}`

Assessment and override operations require Jentic `org:admin`. Local events are
`barrikade.content_flagged`, `barrikade.content_blocked`, `barrikade.scan_failed`, and
`barrikade.override_applied`. A bounded background queue writes metadata-only events and drops
on overflow so event persistence cannot affect execution. They are not registered for Jentic
external product telemetry; Barrikade's database is authoritative.

## Compatibility tests

The contract suite asserts registration precedes every config load, the wrapper receives the
Jentic-provided runner, sync/async paths are protected, exact error types are used, and no
credential/header reaches Core. Disabled overlay tests compare stock responses, streaming,
imports, migrations, and process startup in combined and parts modes.
