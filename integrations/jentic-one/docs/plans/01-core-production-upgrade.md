# Barrikade Core production upgrade

## Responsibility

This document defines the service, package, artifact, runtime, security, and observability
work required before Jentic enforcement can depend on Barrikade.

## Package and dependency layout

The installable service code moves toward `src/barrikade/`. Existing imports from
`barrikade`, `core`, and `models` remain as compatibility re-exports for one release. The
wheel includes the API application; it never depends on a repository working directory.

Dependency groups are separated into:

- fast service runtime: API, Layer A-D inference, persistence, metrics;
- optional Layer E judge runtime;
- training and evaluation;
- development and test.

The production fast image uses Python 3.12 and excludes datasets, notebooks, bundling upload
tools, trainers, `tool_hijacker`, and Layer E. Importing `barrikade` performs no downloads,
model loading, directory creation, logging configuration, or other writes.

`BARRIKADE_*` is the canonical environment prefix. Each former `BARRIKADA_*` name remains an
accepted alias for two minor releases and emits one process-local deprecation warning.

## Artifact contract

An immutable model bundle is baked into the service image or mounted read-only. Its signed
manifest contains bundle version, profile versions, relative artifact paths, sizes, SHA-256
digests, and signing metadata. No absolute path is returned by the API.

Startup order:

1. Parse service configuration and resolve the selected profile.
2. Verify the manifest signature against a configured public key.
3. Reject path traversal, symlinks outside the bundle, size mismatches, and digest mismatch.
4. Load exactly the artifacts required by the profile.
5. Publish readiness with profile and sanitized bundle version.

Any failure leaves liveness healthy and readiness at 503. The error is logged as a closed
reason code; exception text and filesystem paths are not exposed. Serving pods have no
internet egress and never attempt recovery downloads.

## Execution model

Each pod owns one model process. Blocking inference runs in a bounded executor behind a
configurable semaphore and finite admission queue. Admission is deadline-aware:

- an already-expired request returns 504;
- a full queue returns 429 immediately;
- unavailable/unready models return 503;
- a request that exceeds its deadline returns 504;
- shutdown rejects new work, drains admitted work for a bounded grace period, then stops.

The Jentic default is two service replicas, a one-second client deadline, and a fast profile.
Layer E is never loaded in this path.

## Profiles

`jentic_gateway_fast` targets JSON keys and leaves, HTML, text tool output, RFC problem
responses, XML/YAML, and upstream error bodies. `jentic_spec` is separately calibrated for
descriptions, examples, code snippets, defaults, and documentation found in API metadata.

The profile fixes the enabled layers, thresholds, calibration version, text-normalization
limits, and category map. Clients cannot override arbitrary detector settings.

## Detector improvements

Layer A applies strict per-value, aggregate, and recursion limits to candidate encodings.
Successful decoding is evidence of an encoding, not evidence of an attack. Printable decoded
content is rescanned through the complete cascade. Opaque binary/base64 remains benign unless
another signal is present. Diagnostics contain only hashes, sizes, and closed encoding
classifications.

Layers B and C may switch to ONNX only when an evaluation report demonstrates no more than
0.5% verdict disagreement and no regression in false-block performance. A failed parity gate
retains the current backend automatically. Layer D remains the deep fast classifier.

Layer E removes `trust_remote_code`, never exposes raw completions or reasoning, and remains
optional/offline. It cannot enter the inline Jentic profile until it fits the full one-second
budget under load.

## Security and observability

Service-to-service authentication uses a bearer token read from a mounted file. Tokens are
hashed for comparison, scoped to assessment or administration, and never logged. Production
credentials cannot request text diagnostics.

Metrics use bounded labels: profile, verdict, status, deciding layer, category, and bundle
version. They cover admissions, queue rejection, deadlines, duration, readiness, overrides,
and persistence failure. Traces carry request/assessment IDs and digests only. Structured logs
apply an allow-list serializer and never log request models.

## Completion criteria

- Wheel import and API startup succeed outside the repository.
- Production image contains only fast runtime files and has no artifact download path.
- Manifest tampering and missing artifacts prevent readiness.
- Load tests prove bounded queues, 429/504 behavior, and shutdown drain.
- Alias tests cover every renamed setting and the two-release warning policy.
- Artifact and telemetry tests demonstrate no raw content disclosure.
