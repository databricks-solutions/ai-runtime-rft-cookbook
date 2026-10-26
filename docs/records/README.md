# Run records and test results

This folder holds the evidence behind the results in the [README](../../README.md): one folder
per GPU run, and below, the latest result of every test tier.

## Run records

Each run folder holds:

- `qualification.json`: the run record. It lists what the run used (code, recipe, image, model,
  data, evaluation settings, GPUs), the checks it passed, and its status;
- `comparison.json`: the base and trained evaluations, compared question by question;
- `export.json`: the trained checkpoint that was evaluated;
- `calibration.json` (OfficeQA): the judge's results on known cases, before it graded anything;
- `receipts/`: one file per stage, naming the code and image it ran with;
- `SANITIZED.json`: the workspace details replaced by placeholders, and the sha256 of every
  original file.

| Run | Recipe | Status |
|---|---|---|
| [`search-q36-01`](search-q36-01/qualification.json) | `search/verl/qwen3.6-35b-a3b_h100x16` | measured |
| [`oqa-q36-verl-03`](oqa-q36-verl-03/qualification.json) | `officeqa/verl/qwen3.6-35b-a3b_h100x16` | measured |
| [`oqa-q36-miles-03`](oqa-q36-miles-03/qualification.json) | `officeqa/miles/qwen3.6-35b-a3b_h100x16` | measured |
| [`math-2b-02`](math-2b-02/qualification.json) | `math/verl/qwen3.5-2b_h100x8` | measured |
| [`search-q36-smoke-02`](search-q36-smoke-02/qualification.json) | `search/verl/qwen3.6-35b-a3b_h100x16`, `--smoke`, 12 turns | smoke-tested |
| [`math-2b-smoke-01`](math-2b-smoke-01/qualification.json) | `math/verl/qwen3.5-2b_h100x8`, `--smoke` | smoke-tested |
| [`math-2b-smoke-02`](math-2b-smoke-02/qualification.json) | `math/verl/qwen3.5-2b_h100x8`, `--smoke`, image from Unity Catalog | smoke-tested |
| [`oqa-q36-miles-smoke-01`](oqa-q36-miles-smoke-01/qualification.json) | `officeqa/miles/qwen3.6-35b-a3b_h100x16`, `--smoke`, image from Unity Catalog | smoke-tested; [shutdown caveat](oqa-q36-miles-smoke-01/README.md) |

The GLM-5.3-Flash recipes ran before run records existed; their recipe headers describe their
runs.

The older 35B records (`search-q36-01`, `search-q36-smoke-02`, `oqa-q36-verl-03` and
`oqa-q36-miles-03`) have an export hash (`training.weights_sha256`) derived from weight-file
metadata, not their content, so runs of one model share it. Newer records hash the content of
every weight file, and say so (`training.weights_hash: content`).

The OfficeQA verl and search Unity Catalog smoke runs are still in progress. Their records will
be published only after evaluation and qualification finish.

## Test results (2026-10-26)

The [testing tiers](../architecture.md#testing) are described in the architecture. GitHub Actions
isn't enabled on this repository yet. The four CI jobs in
[`ci.yml`](../../.github/workflows/ci.yml) were first replayed on `c627256`, each in a fresh
Python 3.12 environment with the installs the job lists. The CPU results below are the later
rerun on the clean `83b0e80` checkout, reusing those environments and the pinned upstream
sources. This rerun is not an image-level test.

| Tier | What ran | Commit | Result |
|---|---|---|---|
| 1, 2, 4 | CI job `core`: unit and contract tests, every recipe and job, the public-safety scan | 83b0e80 | 616 passed; 3 expected failures (known limits of the vendored math grader) |
| 2, 3, 4 | CI job `verl`: the verl adapter against the pinned verl and a real Qwen3.5 tokenizer | 83b0e80 | 40 passed |
| 2, 4 | CI job `miles`: the miles adapter without miles installed | 83b0e80 | 27 passed, 2 skipped (they need miles) |
| 3 | CI job `miles-api`: the miles adapter against the pinned miles | 83b0e80 | 47 passed |
| 5 | GPU smoke of the math recipe on verl through `databricks air`, with its image from Unity Catalog (`math-2b-smoke-02`) | dba3e87 | smoke-tested |
| 5 | GPU smoke of the search recipe on verl, with 12 turns (`search-q36-smoke-02`) | 2ead10d | smoke-tested |
| 5 | OfficeQA on miles through the recipe runner: two updates, a resumed third, export, evaluation and comparison (`oqa-q36-miles-smoke-01`) | e151733 | smoke-tested record; training shutdown still fails (see the run's note) |
| 6 | Full runs of the four measured recipes (table above) | 7c2a9f1, 593dba3 | measured |

Not covered yet:

- running the CPU tiers on every pull request, which needs GitHub Actions;
- tier 3 inside the framework images (the CPU jobs install the pinned frameworks' Python code
  instead);
- a miles GPU smoke whose training jobs both terminate successfully without no-op retries. The
  published smoke record verifies the checkpoint and evaluations, not clean training shutdown.
