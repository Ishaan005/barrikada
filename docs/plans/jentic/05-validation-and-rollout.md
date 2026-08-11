# Validation and rollout

## Responsibility

This document defines the test pyramid, adversarial evaluation, performance budgets, release
evidence, shadow operation, and production rollout.

## Test suites

### Core unit and contract

Cover v2 validation, closed enums/context, authentication/scopes, constant-time token checks,
idempotency and conflicts, concurrent duplicates, input/queue limits, deadlines, metadata-only
persistence, retention, overrides, artifact verification, readiness, overload, shutdown, and
production diagnostic denial. Export OpenAPI and regenerate the plugin client in CI.

Layer A fixtures cover nested/mixed encodings, Unicode obfuscation, decompression bombs,
recursion/aggregate limits, and benign encoded API data. Backend evaluation enforces the ONNX
0.5% disagreement ceiling plus false-block/latency/recall gates.

### Plugin unit and Jentic contract

Run the Jentic Broker compliance suite and tests for registration-before-config, disabled
short-circuiting, wrapper composition with Jentic's supplied runner, sync/async protection,
content eligibility, bounded JSON/window extraction, compression normalization, every policy
branch, exact exception classes, defensive streaming, event overflow, admin authorization, and
secret/header exclusion.

OpenAPI fixtures exercise every scanned metadata field, canonical digest stability, safe
locators, rollback, flags, blocks, partial/oversize, outages, exact overrides, and invalidation
after any digest/profile/bundle change.

### End to end

Exercise benign JSON/text, malicious keys/leaves, HTML, XML/YAML, RFC problem/error bodies,
binary responses, flags, blocks, scanner timeout/outage, invalid JSON fallback, ignored identity
encoding, malicious specification rollback, and override audit. Run against both combined and
parts deployments.

Disabled image comparison asserts response/status/header equivalence for representative
execution and Registry flows, stock streaming defaults, zero Core calls/events, unchanged
migrations, and successful startup of all process shapes.

## Data-leak testing

Every test run seeds unique canary strings into assessed text, headers, credentials, cookies,
actor emails, prompts, and generated matches. Automated scans of database dumps, structured
logs, traces, metrics, Jentic events, HTTP errors, and readiness/status responses must find no
canary. This is a hard release gate.

## Adversarial evaluation

Maintain separately labeled response and specification corpora. Include direct, indirect,
role/authority manipulation, tool redirection, data exfiltration, encoded/nested, Unicode,
HTML/markup, schema/example, malicious error, and multilingual cases. Benign sets emphasize
API keys as field names, security documentation, code snippets, base64/binary data, policy
descriptions, and quoted attack discussion.

Report recall, precision, false-block and flag rates per surface/profile/category, not only
aggregate. No curated critical attack may be missed; surface recall must be at least 95%,
benign false blocks at most 0.1%, and benign flags at most 1%.

## Performance and resilience

Measure incremental Jentic latency and Core service time at payload buckets up to the 2 MiB
limit, with the release target of at most 200 ms p95 and 750 ms p99 added latency through
64 KiB. Saturation tests verify finite memory, immediate 429 admission rejection, deadline
504s, two-replica disruption tolerance, bounded decompression, event dropping, and graceful
drain. Chaos cases cover Core restart, database loss, DNS failure, token rotation, bundle
tamper, and rolling Jentic upgrades.

## Rollout stages

1. CI-only fixtures and replay evaluation.
2. Internal disabled overlays to prove parity.
3. Shadow assessment with enforcement suppressed by deployment-team policy.
4. Seven consecutive shadow days and at least 10,000 eligible assessments.
5. Review all proposed blocks and representative flags/allows; meet every release gate.
6. Internal enforcement, then limited canary organizations, then graduated production rollout.
7. General availability only after SLO and support review.

Shadow/canary controls are advanced protected chart settings and never expand the public
one-toggle interface.

## Operational signals and rollback triggers

Dashboards show eligible/not-applicable volume, verdict/category/profile rates, latency,
queue saturation, deadlines, failures, bundle versions, overrides, and event drops without
content labels. Alerts cover readiness loss, scanner error/timeout rate, queue rejection,
block-rate shift, persistence failure, and version mismatch.

Automatic rollout pause or manual disable is triggered by a leak canary, missed curated
critical attack, false-block breach, sustained p99 breach, version incompatibility, widespread
503/504, or disabled-parity regression. The immediate safe operation is setting
`barrikade.enabled: false`; incident analysis uses metadata IDs/digests and separately secured
source-system evidence.

## Release evidence packet

Archive the compatibility matrix, exact image digests/SBOMs, OpenAPI/client diff result,
unit/contract/end-to-end reports, adversarial metrics, latency/load results, leak scan, shadow
review totals, known limitations, approval, and rollback rehearsal result for every release.

## Local candidate status (2026-08-11)

The `0.2.0-local.1` candidate proves the packaging and integration path, but it is **not a
production release**. Local evidence currently shows:

- Core, the thin plugin, official Jentic 0.29.1 overlay images, Compose, and Helm build and
  start successfully without modifying Jentic One.
- Disabled overlay construction matches the pinned stock image: the same 44 routes, the same
  route hash, stock streaming enabled, and no Barrikade Broker factory.
- The v2 API passes authentication, idempotency/conflict, limit, override, metadata-only
  persistence, readiness, and overload tests. Canary scans found no assessed text in logs or
  SQLite storage.
- The original Layer B, C, and D ONNX exports each passed a 500-item backend-parity run with
  zero verdict disagreement. This is backend parity evidence, not protected-surface quality
  evidence.
- A Layer D INT8 candidate was rejected: it produced 2.4% verdict disagreement and degraded
  recall. An ORT-optimized FP32 graph preserved verdicts but was slower than the selected
  export, so it was also not promoted.
- The latency release gate fails. Under the local amd64-on-Apple-Silicon container, the
  1 KiB benchmark exceeded 800 ms p95 and 64 KiB requests exceeded the one-second deadline.
  A single 64 KiB diagnostic with a 30-second deadline took approximately 2.2 seconds and
  reached Layer D. Native profiling also remains above the 200 ms target.
- The available legacy evaluation corpus is not sufficient to certify the Jentic response
  and specification false-block/recall gates. A separately labeled Jentic-shaped corpus is
  still required.
- The required seven-day, 10,000-assessment shadow period remains an external staging task.

Consequently, enforcement must remain disabled outside development. The next release work is
to train or distill a smaller calibrated deep classifier for the fast profiles, evaluate it
on Jentic-specific response and specification corpora, rebuild the signed bundle, and repeat
the performance and adversarial gates before shadow deployment.
