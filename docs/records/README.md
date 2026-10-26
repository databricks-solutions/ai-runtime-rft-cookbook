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

The GLM-5.3-Flash recipes ran before run records existed; their recipe headers describe their
runs.

## Test results (2026-10-26)

The [testing tiers](../architecture.md#testing) are described in the architecture. GitHub Actions
isn't enabled on this repository yet, so the four CI jobs in
[`ci.yml`](../../.github/workflows/ci.yml) were run by hand: each in a fresh Python 3.12
environment, on a clean checkout of the commit, with the installs the job lists.

| Tier | What ran | Commit | Result |
|---|---|---|---|
| 1, 2, 4 | CI job `core`: unit and contract tests, every recipe and job, the public-safety scan | 3c92065 | 600 passed; 3 expected failures (known limits of the vendored math grader) |
| 2, 3, 4 | CI job `verl`: the verl adapter against the pinned verl and a real Qwen3.5 tokenizer | 3c92065 | 40 passed |
| 2, 4 | CI job `miles`: the miles adapter without miles installed | 3c92065 | 27 passed, 2 skipped (they need miles) |
| 3 | CI job `miles-api`: the miles adapter against the pinned miles | 3c92065 | 47 passed |
| 5 | GPU smoke of the math recipe on verl (`math-2b-smoke-01`) | c45c5e7 | smoke-tested |
| 5 | GPU smoke of the search recipe on verl, with 12 turns (`search-q36-smoke-02`) | 2ead10d | smoke-tested |
| 5 | GPU smoke of the OfficeQA recipe on miles: two updates, a resumed third, export | 9e2416c | passed (run by hand, before run records existed) |
| 6 | Full runs of the four measured recipes (table above) | 7c2a9f1, 593dba3 | measured |

Not covered yet:

- running the CPU tiers on every pull request, which needs GitHub Actions;
- tier 3 inside the framework images (the CPU jobs install the pinned frameworks' Python code
  instead);
- a GPU smoke on miles through the recipe runner (`--smoke`), with a run record.
