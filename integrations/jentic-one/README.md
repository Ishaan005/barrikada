# Jentic One integration

Barrikade protection is delivered from this repository as a thin Jentic overlay and a private
Core service. Jentic One itself is not patched or modified.

## Where everything lives

This directory is the single home for the Jentic integration:

```text
integrations/jentic-one/
├── src/barrikade_jentic/   # thin runtime plugin
├── deploy/                  # Compose overlay, overlay image, and Helm chart
├── docs/plans/              # architecture and rollout plans
├── evaluation/              # reviewed corpus and source manifests
├── scripts/                 # integration evaluation and model-training tooling
├── examples/notion/         # Jentic-specific example
├── tests/plugin/            # tests against the pinned Jentic package
├── tests/core/              # Core/profile/release-boundary tests
├── openapi/                 # checked-in Core API contract digest
├── pyproject.toml           # thin plugin wheel definition
└── setup.py                 # registration-autoload build hook
```

Only detector and service implementation shared by every integration remains in the repository's
`core/`, `barrikade/`, and `api/` packages. The two Jentic assessment profiles execute there, but
all Jentic adaptation, deployment, evidence, and release ownership is contained here.

## Public configuration

The only Barrikade setting required in normal deployments is:

```yaml
barrikade:
  enabled: true
```

Set it to `false` to use the pinned stock Jentic image and remove the Core service, Core database,
migration job, service token, protected networking, and streaming override. No Barrikade client,
Broker wrapper, scans, or events are created in a disabled Jentic process.

The umbrella chart owns all internal defaults: endpoint discovery, authentication, migrations,
profiles, limits, buffered response handling, and the balanced policy. Operators may replace
generated secrets, PostgreSQL, or the signed model bundle through advanced chart values, but these
are not installation prerequisites.

For local Compose deployments the equivalent switch is:

```text
JENTIC__BARRIKADE__ENABLED=true
```

The supplied overlay then starts the local Core service and metadata store automatically.

## Enabled behavior

Eligible buffered tool responses are assessed before they reach the agent. Definite attacks and
incompletely inspected eligible text are denied; uncertain flags pass with a local security event;
binary responses pass as not applicable; scanner outages fail closed. Streaming is disabled in a
derived in-memory Jentic configuration and cannot bypass the wrapper.

Agent-visible OpenAPI and tool metadata is assessed before Registry ingestion commits. Flags,
blocks, partial scans, and scanner failures reject the full transaction. An administrator can use
an exact assessment/digest/profile/bundle override for an exceptional reviewed specification.

## Administrative endpoints

Combined and Admin/Auth processes expose `/plugins/barrikade/status`, assessment lookup, and exact
specification override routes. Assessment and override operations require Jentic `org:admin`.
They are investigation tools, not setup steps.

Compatibility is initially pinned to Jentic One `0.29.1`. Core, the plugin wheel, chart, and overlay
images share one Barrikade release version.

## Maintainer entry points

- Helm chart: `integrations/jentic-one/deploy/helm`
- Compose overlay: `integrations/jentic-one/deploy/compose.yaml`
- Overlay image: `integrations/jentic-one/deploy/overlay.Dockerfile`
- Implementation plans: `integrations/jentic-one/docs/plans`
- Reviewed security corpus: `integrations/jentic-one/evaluation/corpora/jentic-v1`

These are maintainer paths only. Normal operators still use the single `barrikade.enabled` toggle
and do not run plugin, model, or migration commands.
