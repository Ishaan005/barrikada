# Jentic One integration

Barrikade protection is delivered from this repository as a thin Jentic overlay and a private
Core service. Jentic One itself is not patched or modified.

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
