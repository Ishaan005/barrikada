# Barrikade umbrella chart for Jentic One

This chart keeps the normal integration surface to one value:

```yaml
barrikade:
  enabled: true
```

When enabled, the chart selects the Barrikade Jentic overlay image, installs two Core replicas,
runs Jentic and Core migrations, provisions service authentication, supplies protected internal
configuration, and applies network policy. It supports `jentic.mode: combined` (control plane plus
a separate Broker) and `jentic.mode: parts`.

When disabled, it selects the immutable stock Jentic `0.29.1` image and renders no Barrikade Core,
database, token, migration, mount, environment setting, or network policy.

Advanced values cover externally managed tokens/databases and read-only model bundle mounts. An
external database also supplies its allowed egress CIDRs, keeping the serving pod restricted to DNS
and PostgreSQL. The published Core release image already includes a signed fast-profile bundle, so
these replacements are optional and do not change the public setup flow.
