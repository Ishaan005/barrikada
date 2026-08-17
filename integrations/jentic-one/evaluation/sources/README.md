# Jentic corpus sources

`jentic-corpus-sources.json` is the approved input manifest for building a draft, human-review
pool. Every downloaded asset is pinned to a full commit and SHA-256 digest. The source data and
generated review pool live under `build/`, are ignored by Git, and are not model-training or
release evidence.

## Approved inputs

| Source | License | Use in draft pool | Qualification |
|---|---|---|---|
| [NVIDIA Agentic IPI v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1) | CC-BY-4.0 | Runtime and specification attacks | 1,272 agentic indirect-injection records; 676 remain after normalized deduplication and redaction. |
| [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | MIT | Runtime benign values; runtime and specification attacks | 1,054 injected tool-response cases and 2,347 simulated response entries. The builder selects deterministic, deduplicated subsets. |
| [Stripe OpenAPI](https://github.com/stripe/openapi) | MIT | Runtime benign fixtures and specification benign metadata | Official API description and fixture data. |
| [GitHub REST API description](https://github.com/github/rest-api-description) | MIT | Specification benign metadata | Official, large real-world API description. |
| [OpenAI OpenAPI](https://github.com/openai/openai-openapi) | MIT | Specification benign metadata | Official API description with modern AI API terminology. |
| [Jentic One OpenAPI](https://github.com/jentic/jentic-one) | Apache-2.0 | Specification benign metadata | Exact pinned target-product API surface; Jentic One remains unmodified. |

The existing `jentic-training-sources.json` separately pins the MIT-licensed
[S-Labs prompt-injection dataset](https://huggingface.co/datasets/S-Labs/prompt-injection-dataset).
It also pins the MIT-licensed [AgentDojo](https://github.com/ethz-spylab/agentdojo) archive used
for independent agentic-attack training. These sources are for training/validation input only
and are never used as the independent Jentic release test corpus.

## Held or rejected

- Microsoft BIPIA is held for legal review because its benchmark components have mixed licenses
  and separate source-download terms.
- APIs.guru is held because aggregated API descriptions do not have one guaranteed license.
- IPIBench and the reviewed community dataset are excluded because a sufficiently clear license
  and production-training provenance statement was not found.

## Reproducible developer workflow

These are Barrikade development commands, not Jentic installation steps:

```shell
python integrations/jentic-one/scripts/evaluation/acquire_jentic_corpus_sources.py
python integrations/jentic-one/scripts/evaluation/build_jentic_review_pool.py
```

Acquisition fails on an unexpected digest, unsafe path, unpinned revision, non-HTTPS URL, or
unapproved source. The builder removes common email, phone, SSN, credential, private-key and
synthetic subject-ID shapes before validation. Reports contain only counts, IDs and hashes.
