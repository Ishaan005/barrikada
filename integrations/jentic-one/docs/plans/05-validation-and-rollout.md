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
benign false blocks at most 0.3%, and benign flags at most 1%.

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
3. Rehearse the single-toggle rollback and confirm partner opt-in.
4. Begin monitored partner GA with balanced enforcement and review every block.
5. Accumulate seven shadow-equivalent days and at least 10,000 eligible assessments in live
   operation before expanding beyond design partners.
6. Review representative flags/allows, latency, outages and false blocks before each expansion.
7. Broad self-serve availability follows SLO and support review.

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

### Fast-model iteration (2026-08-11)

The next model-readiness iteration added a closed, versioned Jentic corpus format, leakage
validation, content-free evaluation reports, pinned training-source metadata, and a compact
student training/calibration pipeline. The initial seed contains 48 draft cases balanced across
runtime responses, specification metadata, benign hard negatives, and injections. It is an
engineering seed, not statistically sufficient or human-reviewed release evidence.

The baseline and three candidate iterations were evaluated against that independent seed:

- The current cascade reached 100% runtime attack recall and 91.7% specification attack recall,
  but produced 33.3% runtime benign blocks, 33.3% runtime benign flags, and 83.3% specification
  benign flags.
- The locally cached Prompt Guard 2 86M candidate was fast on short inputs (about 33--37 ms
  p95) but reached only 75% runtime and 41.7% specification attack recall and missed three
  critical cases. It was rejected.
- A first synthetic-only 67M-parameter DistilBERT student ran in about 9--12 ms p95 but reached
  only 75% runtime and 50% specification attack recall and missed three critical cases. It was
  rejected.
- A 67M-parameter DistilBERT student trained from pinned MIT-licensed train/validation data plus
  synthetic Jentic-shaped training examples ran in about 8--9 ms p95 on short seed inputs. Its
  independent quality result still failed: 66.7% runtime benign blocks, 58.3% specification
  benign blocks, and 83.3% specification attack recall. It was rejected and was not added to
  the runtime bundle.

The student validation set itself could not meet the combined 95% attack-recall and 0.1%
false-block constraints at any calibrated threshold. No threshold or model artifact is promoted
from this iteration. The next data task is independent human review and expansion of
Jentic-shaped benign hard negatives and indirect attacks; the next model task should use those
training/validation families while keeping the frozen test families isolated.

### Data-source acquisition and draft pool (2026-08-11)

The next data milestone is now mechanically unblocked. A checked-in source manifest pins ten
assets from six approved sources by full commit, license and SHA-256. A content-safe acquisition
tool verifies all 36,462,037 downloaded bytes, and a deterministic builder creates a redacted,
schema-valid draft review pool without changing Jentic One or committing source data.

The resulting 5,552 candidates comprise 1,200 runtime benign cases, 1,176 runtime attacks,
2,000 specification benign cases and 1,176 specification attacks. Attack material comes from
NVIDIA's agentic indirect-injection set and InjecAgent's injected tool responses. Benign material
comes from InjecAgent simulated responses plus official Stripe, GitHub, OpenAI and Jentic API
descriptions and Stripe fixtures. The build redacted email, phone, SSN, credential/private-key
and synthetic subject-ID shapes before corpus validation.

This completes source discovery and reproducible acquisition, not data approval. Every generated
row is deliberately `draft`; it cannot satisfy a release gate. The next controlled step is human
security review, semantic-family consolidation, attribution review, and a leakage-safe split.
Only the frozen reviewed test partition may be used to judge the 95% recall and benign-error
gates. BIPIA and APIs.guru remain on legal hold, while sources without sufficiently clear license
or provenance were excluded.

### Approved corpus and candidate outcome (2026-08-11)

The project owner approved all 5,552 redacted candidates. A deterministic family-stratified
freeze now reserves 4,000 rows for the independent test partition: exactly 1,000 benign and 1,000
attack cases on each protected surface. The remaining 1,552 rows are calibration-only validation
data. No reviewed row enters training, and the freeze manifest records zero normalized-text
overlap with the 13,480-row v2 student dataset.

The old v2 student was rejected on the approved validation partition. It preserved roughly 75%
attack recall at its best surface-aware threshold but produced 11.8%--22.5% benign blocks. A v3
iteration added 4,000 non-held-out benign tool/spec examples; it over-corrected and reduced attack
recall to 67.6%--73.3%, so it was rejected without opening the final test partition.

The v4 iteration added a pinned MIT-licensed AgentDojo archive. It yielded 629 unique redacted
agentic injections and 2,516 contextual attack-training rows, all disjoint from the reviewed
corpus. The standalone v4 Layer D reached 99.43% validation recall at about 15 ms p95, but benign
false blocks remained 1.5% runtime and 3.5% specification. The full fast cascade reached 96.02%
runtime and 97.73% specification recall at under 70 ms p95, while false blocks worsened to 5.5%
and 6.8%. The candidate is therefore rejected and the 4,000-case final test remains sealed.

The next model phase must calibrate or replace Layers B and C for the two Jentic profiles and
improve short benign metadata handling before another Layer D candidate is admitted. Threshold
changes alone cannot satisfy the validation frontier. No candidate artifact is promoted to the
signed runtime bundle.

### Jentic profile routing and v5 sealed-test result (2026-08-11)

Error attribution showed that the legacy Layer B and C decisions caused every cascade attack
miss and most false blocks on the approved Jentic validation data. The Jentic-only fast profiles
now route Layer A output directly to Layer D; other Barrikade profiles retain the existing
cascade. Short JSON keys and specification titles, summaries, and tags may return at Layer A
only when they are bounded structural labels without agent-control cues. Runtime values,
descriptions, defaults, examples, and external documentation still reach Layer D. The plugin
preserves these source subtypes in the v2 assessment contract.

The v5 candidate added 19,965 real benign Jentic-shaped training rows and retained the 2,516
pinned AgentDojo attack rows. Its 35,961-row source file has zero normalized-text overlap with
the reviewed corpus. With validation-frozen thresholds of 0.919/0.920 and the Jentic routing
policy, the 1,552-row validation partition passed every quality gate: 99.43% recall on both
surfaces, zero benign blocks and flags, no critical misses, and 18.7--22.0 ms p95 latency.

That result authorized the single planned opening of the 4,000-row sealed test partition. The
test result reached 98.7% runtime recall and 99.3% specification recall, with no critical misses,
zero runtime benign blocks, and approximately 20.6--20.9 ms p95 latency. Specification benign
false blocks were 0.3%, exceeding the 0.1% release gate. Consequently v5 is rejected, its model
and thresholds are not promoted to the signed runtime bundle, and no post-test tuning may use
the exposed test examples. A future candidate requires new training data and a newly reviewed,
independently frozen test partition before a release decision.

### Founder risk acceptance and partner GA policy (2026-08-11)

The project owner subsequently accepted a 0.3% benign false-block rate for initial GA, based on
Barrikade's startup stage and the need to validate with design partners. Policy version
`jentic-partner-ga-2026-08-11` therefore supersedes the earlier 0.1% gate. This is explicitly a
post-test business-risk decision, not a claim that v5 met the original preregistered threshold.
The sealed results and original rejection remain recorded above.

Under the approved policy, v5 satisfies the adversarial quality gate and may advance to a
monitored partner GA candidate: runtime/specification recall remain 98.7%/99.3%, specification
false blocks equal the accepted 0.3% ceiling, runtime false blocks are zero, no critical attack
is missed, and native classifier latency is within budget. Partner GA requires explicit opt-in,
review of every block, operational alerting, and a rehearsed `barrikade.enabled: false` rollback.
The seven-day/10,000-assessment target becomes a scale-up gate rather than a prerequisite for
the first partner release.

The v5 INT8 ONNX artifact has also cleared production-format qualification on all 4,000 test
cases. It produced 0.125% verdict disagreement against PyTorch, unchanged aggregate false blocks,
99.15% aggregate recall, and improved p95 latency from 18.3 ms to 14.8 ms. Surface-level pipeline
evaluation of that exact artifact passed the partner-GA policy with 99.0% runtime recall, 99.3%
specification recall, 0% runtime false blocks, 0.3% specification false blocks, no critical
misses, and approximately 14.7 ms p95. The calibration metadata is included in the signed
`0.2.0` bundle.

Local Core and Jentic overlay images were built and smoke-tested with the signed bundle. Core
readiness reported the expected profile and bundle version; a benign assessment was allowed and
a direct injection was blocked. Helm lint and enabled rendering pass. These are local release
artifacts only: publishing immutable registry digests and deploying to a partner environment are
separate externally visible release actions.
