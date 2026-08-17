# Jentic One integration overview and milestones

Status: implementation baseline  
Initial compatibility target: Jentic One 0.29.1  
Repository: this Barrikade Core repository only

## Outcome

Barrikade protects Jentic tool responses and imported OpenAPI metadata without changing
Jentic source code. The public installation surface is one value:

```yaml
barrikade:
  enabled: true
```

The repository publishes one coordinated version of four artifacts: the Barrikade service
image, the `barrikade-jentic` wheel, Jentic overlay images, and the umbrella Helm chart.
The Compose overlay uses the same version and configuration contract.

## Invariants

1. Jentic One remains an unmodified, pinned upstream dependency.
2. Barrikade Core and all integration code remain in this repository.
3. Disabled mode follows stock Jentic construction and performs no Barrikade I/O.
4. Raw assessed content never enters Barrikade persistence, logs, traces, events, or errors.
5. Eligible content that cannot be fully inspected fails closed while protection is enabled.
6. The plugin contains no detector, model, training, or artifact dependency.
7. Normal users run no plugin, download, migration, or bootstrap command.

## Work breakdown and dependency order

| Milestone | Deliverable | Depends on | Exit evidence |
|---|---|---|---|
| M0 | Plans and frozen contracts | none | six reviewed plan documents |
| M1 | Production Core baseline | M0 | Python 3.12 image, import-safe package, verified bundle, bounded inference |
| M2 | Assessment API v2 | M1 | OpenAPI contract, auth, limits, idempotency, metadata-only persistence |
| M3 | Thin plugin skeleton | M2 | wheel dependency audit, config autoload, disabled parity |
| M4 | Runtime response enforcement | M3 | Broker compliance and policy-branch tests |
| M5 | Registry ingest enforcement | M3 | transaction rollback and exact-override tests |
| M6 | Automatic deployment | M4, M5 | combined/parts Helm and Compose smoke tests |
| M7 | Partner GA and monitored rollout | M6 | opt-in partners, rollback rehearsal and live review |

Each milestone is independently releasable behind `barrikade.enabled: false`. M4 and M5
must not be generally enabled until M7 gates pass.

## Compatibility matrix

| Component | Supported initially | Compatibility rule |
|---|---|---|
| Jentic One | `>=0.29.1,<0.30` | exact overlay base-image digest per release |
| Python | 3.12 | Core and plugin test on the same minor version |
| Barrikade API | v2 | generated client must match checked-in OpenAPI |
| PostgreSQL | 16+ | production metadata store |
| SQLite | 3 | local Compose and tests only |
| Kubernetes | 1.28+ | NetworkPolicy and batch/v1 Jobs required |
| Helm | 3.13+ | umbrella chart owns conditional resources |

A Jentic minor-version update is blocked until the plugin contract suite passes against the
new official images. The initial plugin upper bound deliberately prevents an unnoticed 0.30
upgrade.

## Release gates

All gates are mandatory:

- At least 95% recall on each protected surface: JSON/text response, HTML/error response,
  and OpenAPI metadata.
- No curated critical attack is missed.
- Benign false blocks are at most 0.3%; benign flags are at most 1%.
- Added latency is at most 200 ms p95 and 750 ms p99 for payloads up to 64 KiB.
- No raw content appears in database rows, logs, traces, Jentic events, or errors.
- Disabled overlay behavior is equivalent to the pinned stock Jentic image.
- Core handles queue saturation, deadlines, artifact failure, and graceful shutdown as
  specified.
- Initial partner GA may begin with enforcement enabled after rollback rehearsal and explicit
  partner opt-in. Seven shadow days and 10,000 eligible assessments remain the scale-up gate
  before broad self-serve availability.
- Every proposed block plus a representative sample of flags and allows is reviewed.

## Release and rollback

Service, plugin, overlays, chart, and generated client share a release version. Deployment
rolls Core first, waits for the new bundle to become ready, then rolls Jentic overlays. CI
rejects a release when the generated-client contract hash differs from Core's v2 OpenAPI.

Operational rollback is the single value `barrikade.enabled: false`. It restores official
Jentic behavior, removes the Broker wrapper and ingest enforcement, leaves audit metadata in
place, and scales/removes Core and its migration Job. No database downgrade is required.

## Deferred scope

Outbound request scanning, session intent, drift, risk budgets, SSE, indefinite streaming,
and inline Layer E adjudication are intentionally outside the first release.
