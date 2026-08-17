# Deployment and user experience

## Responsibility

This document defines the one-toggle experience, build artifacts, secret wiring, automated
migrations, upgrades, and rollback for Helm and Compose.

## Public configuration

The standard values file documents only:

```yaml
barrikade:
  enabled: false
```

Changing it to `true` is sufficient in the Barrikade umbrella chart. Operators may override
advanced values, but installation instructions never require endpoint, token, profile, model,
database, timeout, queue, streaming, migration, or bootstrap configuration.

## Helm behavior

When enabled, the umbrella chart:

- substitutes Barrikade overlay images built from pinned official Jentic 0.29.1 image digests;
- deploys two Core replicas with anti-affinity, readiness, bounded resources, and rolling update;
- runs idempotent Core migrations automatically through pod initialization and a
  release-versioned migration Job;
- creates a release-scoped service token Secret unless an external Secret is referenced;
- mounts the token as a file and injects a fixed cluster-local Core endpoint into protected pods;
- supplies gateway/spec profiles, one-second timeout, 2 MiB text limit, event queue size,
  balanced enforcement, and bundle version;
- disables Jentic streaming internally through the plugin bootstrap;
- applies least-privilege NetworkPolicies between protected Jentic pods, Core, DNS, and the
  external database;
- handles both combined and parts values.

When disabled, official Jentic images and commands are used, Core and its migration Job are
not rendered, Barrikade secrets/config are not mounted, and no streaming value is changed.
The database is retained unless an operator separately applies its retention policy.

Generated credentials use a chart-owned Kubernetes lookup so Helm upgrades reuse the same
value. Templates never emit a token in notes or labels. Production installations should point
to an externally managed Secret.

## Overlay images

One shared overlay is built from the pinned official Jentic 0.29.1 application image and used
for each process role through `JENTIC__APPS`. Its multi-stage build adds only the
`barrikade-jentic` wheel, registration autoload, and internal bootstrap command. It does not copy
Barrikade Core, model artifacts, compilers, package caches, or source trees. The image runs under
the upstream user.

The official base image is pinned by immutable digest and built in CI. All images share the
Barrikade release version label and declare Jentic/API compatibility labels. SBOM, provenance,
and vulnerability evidence remain release gates.

## Core service image

The fast service image uses Python 3.12, a non-root runtime, a read-only root filesystem,
dropped capabilities, and a verified read-only model bundle. It contains the API and database
migration executable. It has no download credential, cloud SDK, shell requirement, training
code, or Layer E dependency.

## Compose behavior

`integrations/jentic-one/deploy/compose.yaml` is a single overlay. The public switch is:

```text
JENTIC__BARRIKADE__ENABLED=true
```

The overlay selects the Barrikade Jentic image and automatically starts Core plus a local
SQLite metadata volume. Core migration is part of service startup coordination, not a user
command. With the switch false, the plugin delegates to stock Jentic construction and the Core
entry point exits before starting a server.

## Upgrade and rollback

Chart releases pin all coordinated artifact versions. An upgrade runs forward-only Core
migrations, rolls Core and verifies the selected bundle, then rolls Jentic. API and bundle
compatibility appear on the plugin status endpoint. A rollout halts on readiness failure.

Emergency rollback sets `barrikade.enabled: false`. This removes enforcement without requiring
a plugin command, database rollback, model download, or Jentic migration. Re-enabling uses the
same retained audit database and release-scoped secret.

## Acceptance criteria

- Fresh and upgrade installs require only the single public value.
- Render tests prove disabled charts contain no Barrikade workload, secret, mount, endpoint, or
  streaming mutation.
- Combined and parts smoke tests include migrations, readiness, execution, import, and rollback.
- Overlay filesystem/dependency audits prove no Core/model dependency crossed the boundary.
- Token rotation, external Secret, interrupted migration, and Core-not-ready scenarios are tested.
