# Records

The evidence behind the support matrix: the sanitized qualification records of GPU runs, one
directory per run, and the test runs of each testing tier ([architecture](../architecture.md#testing)).
In each run directory `SANITIZED.json` lists the text replaced by placeholders and the sha256 of
every original file.

## Tests by tier (2026-10-26)

GitHub Actions does not run on this repository yet, so the four jobs of
[`ci.yml`](../../.github/workflows/ci.yml) were replayed outside it: each job in a fresh Python 3.12
virtual environment on a clean checkout of the commit, with the installs the job lists.

| Tier | What ran | Commit | Result |
|---|---|---|---|
| 1, 2, 4 | CI job `core`: the core's unit and contract tests, every recipe and checked-in job, the public-safety scan | 3c92065 | 600 passed; 3 expected failures (strict xfails: known limits of the vendored math grader) |
| 2, 3, 4 | CI job `verl`: the verl adapter against the pinned verl (8718ca30) and a real Qwen3.5 tokenizer | 3c92065 | 40 passed |
| 2, 4 | CI job `miles`: the miles adapter without miles | 3c92065 | 27 passed, 2 skipped (they need miles) |
| 3 | CI job `miles-api`: the miles adapter against the pinned miles (its tarball, sha256-checked) | 3c92065 | 47 passed |
| 5 | verl: the recipe runner's `--smoke` of `math/verl/qwen3.5-2b_h100x8`, run math-2b-smoke-01 (2026-10-25) | c45c5e7 | smoke-tested ([record](math-2b-smoke-01/qualification.json)) |
| 5 | verl: the recipe runner's `--smoke` of `search/verl/qwen3.6-35b-a3b_h100x16` with 12 turns, run search-q36-smoke-02 (2026-10-26) | 2ead10d | smoke-tested ([record](search-q36-smoke-02/qualification.json)) |
| 5 | miles: two updates, a resumed third, retention, attestations and export of `officeqa/miles/qwen3.6-35b-a3b_h100x16`, run oqa-q36-miles-phase2-smoke (2026-10-25) | 9e2416c | passed; from before the records and the recipe runner, so it has no record |
| 6 | `math/verl/qwen3.5-2b_h100x8`, run math-2b-02 (2026-10-25) | 7c2a9f1 | measured ([record](math-2b-02/qualification.json)) |
| 6 | `officeqa/verl/qwen3.6-35b-a3b_h100x16`, run oqa-q36-verl-03 (2026-10-25) | 7c2a9f1 | measured ([record](oqa-q36-verl-03/qualification.json)) |
| 6 | `officeqa/miles/qwen3.6-35b-a3b_h100x16`, run oqa-q36-miles-03 (2026-10-25) | 7c2a9f1 | measured ([record](oqa-q36-miles-03/qualification.json)) |
| 6 | `search/verl/qwen3.6-35b-a3b_h100x16`, run search-q36-01 (2026-10-26) | 593dba3 | measured ([record](search-q36-01/qualification.json)) |

The GLM-5.3-Flash recipes' GPU runs predate the records; their recipe headers list them.

Not covered yet: the CPU tiers on every pull request (they need GitHub Actions), tier 3 inside the
framework images (the replay installs the pinned frameworks' Python sources on CPU), and a
miles GPU smoke run by the recipe runner, with its record.
