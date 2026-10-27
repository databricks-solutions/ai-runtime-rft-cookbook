# Results and test coverage

The evaluations below compare a base model with its trained checkpoint on the same held-out
questions. The links open the machine-readable evidence: configuration, checks, and scores.
For setup and commands, start with the [cookbook README](../../README.md).

## Evaluated recipes

| Task and model | Framework | Before → after | Change, 95% CI | Evidence |
|---|---|---|---|---|
| Agentic search · Qwen3.6-35B-A3B | verl | 28.2% → 37.4% | +9.2 points (+5.2 to +13.4) | [Qualification](search-q36-01/qualification.json) · [Comparison](search-q36-01/comparison.json) |
| OfficeQA · Qwen3.6-35B-A3B | verl | 20.3% → 23.8% | +3.5 points (−2.9 to +9.9) | [Qualification](oqa-q36-verl-03/qualification.json) · [Comparison](oqa-q36-verl-03/comparison.json) |
| OfficeQA · Qwen3.6-35B-A3B | miles | 20.3% → 23.8% | +3.5 points (−3.5 to +11.6) | [Qualification](oqa-q36-miles-03/qualification.json) · [Comparison](oqa-q36-miles-03/comparison.json) |
| Math · Qwen3.5-2B | verl | 61.8% → 62.0% | +0.2 points (−2.8 to +3.2) | [Qualification](math-2b-02/qualification.json) · [Comparison](math-2b-02/comparison.json) |

Search uses answer exact match. Math checks MATH-500 answers. OfficeQA uses supported pass@1:
a correct answer counts only when the judge also accepts its cited evidence. The OfficeQA
intervals include zero; the math recipe is a two-update workflow test, not a quality benchmark.

Confidence intervals use paired bootstrap resampling over questions. Training probes are
reported separately from held-out questions and do not enter the held-out scores.

The search index contains the held-out questions' source passages and distractors. Base and
trained models search the same corpus; the result measures question answering over that corpus,
not generalization to unseen documents.

## Integration tests

GPU smoke tests exercise short training, restart/resume, export, and evaluation. They test the
workflow rather than a useful training budget.

| Recipe | Coverage | Evidence |
|---|---|---|
| Math on verl | Complete smoke test through `databricks air`, using a Unity Catalog image | [Qualification](math-2b-smoke-02/qualification.json) |
| Agentic search on verl | Complete smoke test with the recipe's 16-turn budget and a Unity Catalog image | [Qualification](search-q36-smoke-03/qualification.json) |
| OfficeQA on miles | Checkpoint, resume, export, and evaluation verified; training shutdown remains unreliable | [Qualification](oqa-q36-miles-smoke-01/qualification.json) · [Limitation](oqa-q36-miles-smoke-01/README.md) |

The CPU suites cover agent decisions, tools, scorers, evaluation accounting, checkpoint storage,
and recipe/job consistency. Adapter suites also test hooks against pinned framework code.
These tests do not replace GPU testing inside the framework images. See
[Testing](../architecture.md#testing) for the scope and [the CI workflow](../../.github/workflows/ci.yml)
for the environments and commands.

## Reading the evidence

Each published evaluation folder contains:

- `qualification.json`: model, data, recipe, code and image identities; checks and final status;
- `comparison.json`: paired scores, confidence intervals, and per-split results;
- `export.json`: the checkpoint and exported weight-file identity;
- `calibration.json`, for judged tasks: the judge's results on known examples;
- `receipts/`: the code, image, and overrides used by each stage;
- `SANITIZED.json`: original file hashes and the workspace details replaced by placeholders.

A `measured` qualification describes the full recipe; `smoke-tested` describes a modified
configuration, such as a shorter training budget. Read the checks and configuration alongside
the status. A qualification record does not establish that every job terminated cleanly.

The older 35B evaluations above and the 12-turn search smoke identify exported weights using
file metadata, not content. Newer exports carry `training.weights_hash: content`. Historical
records are preserved as produced, rather than rewritten to claim stronger checks.
