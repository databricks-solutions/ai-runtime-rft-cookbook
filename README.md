# AI Runtime RFT Cookbook

Reinforcement fine-tuning (RFT) of tool-using agents on Databricks AI Runtime, with more than one
RL framework. An agentic task (tools, prompt, decision rules, scorer) is written once; each
framework runs it through a thin adapter on its own extension points, with no fork. A recipe pins
one task to one framework, model and GPU layout. Every training and evaluation episode produces
the same `EpisodeRecord`, so scoring, metrics and evidence are the same everywhere.

Design: [docs/architecture.md](docs/architecture.md).

## Support matrix

| Task | Framework | Model | Recipe | Status |
|---|---|---|---|---|
| Math with a calculator (quickstart) | verl 0.10.0.dev (8718ca30) | Qwen3.5-2B, 1×8 H100 | [`math/verl/qwen3.5-2b_h100x8`](recipes/math/verl/qwen3.5-2b_h100x8/recipe.yaml) | measured (run math-2b-02, 2026-10-25, [record](docs/records/math-2b-02/qualification.json)): every stage passed on the committed code; MATH-500 61.8% → 62.0%, no change, as expected for a plumbing recipe; the recipe runner's GPU smoke (`--smoke`) is smoke-tested on it (run math-2b-smoke-01, [record](docs/records/math-2b-smoke-01/qualification.json)) |
| OfficeQA | verl 0.10.0.dev (8718ca30) | Qwen3.5-35B-A3B, 2×8 H100 (8 train, 8 serve), hosted GLM-5.3-Flash judge | [`officeqa/verl/qwen3.5-35b-a3b_h100x16`](recipes/officeqa/verl/qwen3.5-35b-a3b_h100x16/recipe.yaml) | superseded (OfficeQA continues on Qwen3.6, the rows below; not rerun on the final code): parity passed (with the reference's tool-schema view); two training seeds, held-out supported pass@1 15.1% → 22.1% (+7.0 points, 95% interval [+1.2, +13.4]) and → 20.9% (+5.8 [0.0, +12.2]) |
| Math with a calculator (technical smoke) | miles (upstream v0.1.3) | GLM-5.3-Flash 4-layer slice, 1×8 B300 | [`math/miles/glm-5.3-flash-4layer_b300x8`](recipes/math/miles/glm-5.3-flash-4layer_b300x8/recipe.yaml) | done: GPU smoke passed (2 rollouts, 2 updates, checkpoint published) |
| OfficeQA | verl 0.10.0.dev (8718ca30) | Qwen3.6-35B-A3B, 2×8 H100 (8 train, 8 serve), hosted GLM-5.3-Flash judge | [`officeqa/verl/qwen3.6-35b-a3b_h100x16`](recipes/officeqa/verl/qwen3.6-35b-a3b_h100x16/recipe.yaml) | measured (the reference OfficeQA recipe; run oqa-q36-verl-03, 2026-10-25, [record](docs/records/oqa-q36-verl-03/qualification.json)): 100 updates, held-out supported pass@1 20.3% → 23.8% (+3.5 points, 95% interval [−2.9, +9.9]); two earlier runs +2.9 and +0.0: about three points, which no single run resolves |
| OfficeQA | miles (upstream v0.1.3) | Qwen3.6-35B-A3B, 2×8 H100 (8 train, 8 serve), matched to the verl recipe and compared with its baseline | [`officeqa/miles/qwen3.6-35b-a3b_h100x16`](recipes/officeqa/miles/qwen3.6-35b-a3b_h100x16/recipe.yaml) | measured (run oqa-q36-miles-03, 2026-10-25, [record](docs/records/oqa-q36-miles-03/qualification.json)): 100 updates, held-out supported pass@1 20.3% → 23.8% (+3.5 points, 95% interval [−3.5, +11.6]) against the measured verl run's baseline, where verl also reached 23.8%; an earlier run with the MoE load-balancing loss on (verl has it off) regressed to 11.0% |
| Math with a calculator (training trial) | miles (upstream v0.1.3) | GLM-5.3-Flash, 4×8 B300 | [`math/miles/glm-5.3-flash_b300x32`](recipes/math/miles/glm-5.3-flash_b300x32/recipe.yaml) | done: trained 3 updates and published a full resumable checkpoint (4.8 TB) |
| OfficeQA (long-context measurement) | miles (upstream v0.1.3) | GLM-5.3-Flash, 8×8 B300 | [`officeqa/miles/glm-5.3-flash_b300x64`](recipes/officeqa/miles/glm-5.3-flash_b300x64/recipe.yaml) | done, accepted without a completed run here: the full model trains at 32 GPUs (row above) and has trained on OfficeQA before; the two 64-GPU attempts here stopped in the first update (see the recipe header) |
| Agentic search (MuSiQue, multi-hop) | verl 0.10.0.dev (8718ca30) | Qwen3.6-35B-A3B, 2×8 H100 (8 train, 8 serve), Databricks Vector Search, exact-match reward | [`search/verl/qwen3.6-35b-a3b_h100x16`](recipes/search/verl/qwen3.6-35b-a3b_h100x16/recipe.yaml) | smoke-tested (run search-q36-smoke-02, 2026-10-26, [record](docs/records/search-q36-smoke-02/qualification.json)): 3 updates; held-out exact match 25.4% → 30.2%, mostly because the model stops searching sooner: half the base model's episodes reach the 12-turn limit (254 of 500; 172 after training), so the recipe now allows 16. The full run is not done. An earlier prototype (Qwen3.5-35B-A3B, verl 0.9, the same held-out questions and index) raised held-out exact match from 34.8% to 37.6% and 40.4% (two seeds) |

Recipe paths are under `recipes/`. *measured*: a run of the recipe as checked in passed every check
of its qualification record, which its `compare` stage writes (see
[Evidence and qualification](docs/architecture.md#evidence-and-qualification); the published
records are in [`docs/records`](docs/records)); *smoke-tested*: a
reduced run did. *done*: the recipe's GPU runs passed, as its header records, on code from before the
records (the GLM-5.3-Flash OfficeQA measurement is the exception the row states). *untested*: the code
is complete and passes its CPU tests (including against the pinned verl), but its GPU runs are not
complete. *superseded*: kept with its results but no longer run; the recipe for the newer model replaces
it.

## Layout

```
src/rft_cookbook/    framework-neutral: agent (episode format, shared policy, Chat Completions
                     harness), tools, scoring and judges, tasks, checkpoints, evidence, evaluation
frameworks/verl/     the verl adapter (rft_verl), its job launcher and its image
frameworks/miles/    the upstream-miles adapter (rft_miles), its job launcher, model launchers and image
recipes/             the generic agentic recipe per framework, and concrete recipes
tests/               CPU tests of the core; each adapter's tests are in frameworks/<name>/tests
```

A verl user installs and imports nothing from other frameworks.

## Quickstart: math on verl

1. Build the verl image, or use one built from `frameworks/verl/image` (see
   [frameworks/verl/README.md](frameworks/verl/README.md)).
2. Copy `site.example.env` to `site.env` (gitignored) and fill in your workspace profile,
   volume root, image and Hugging Face token secret.
3. Submit the recipe's jobs in order (see [Running a recipe](#running-a-recipe)), waiting for each to finish:
   `prep`, `stage_model`, `baseline_eval`, `train`, `export`, `eval`, `compare`.

## Running a recipe

Every recipe has one AI Runtime job per stage, checked in next to it:
`recipes/<task>/<framework>/<recipe>/jobs/<stage>.yaml`. A job names its recipe and leaves
placeholders for what belongs to your workspace and your run; fill them in at submission:

```bash
source site.env
air run -f recipes/math/verl/qwen3.5-2b_h100x8/jobs/prep.yaml -p "$RFT_PROFILE" --override \
  env_variables.RFT_RUN_ID=math-2b-01 env_variables.RFT_VOLUME_ROOT="$RFT_VOLUME_ROOT" \
  environment.docker_image.url="$RFT_VERL_IMAGE" secrets.HF_TOKEN="$RFT_HF_TOKEN_SECRET"
```

`secrets.HF_TOKEN` is needed only by the stages that download from the Hugging Face Hub (their
jobs list it). Agentic search also needs `env_variables.RFT_VS_INDEX`, the Vector Search
index its episodes search, in the stages that run episodes (`baseline_eval`, `train`, `eval`). All stages of one run use the same run id; each finds its inputs on the volume
under the run's directory. Inside the job the launcher resolves the recipe from the code snapshot,
so a recipe change applies to the next job submitted.

The `compare` stage also writes the run's qualification record (`qualification.json` in the run's
directory): the code, recipe, image, model, data and evaluation it ran with, the checks it passed,
and the status that follows (`measured`, `smoke-tested` or `failed`). Pass
`env_variables.RFT_CODE_COMMIT=<commit>` and `env_variables.RFT_IMAGE=<image>` to every stage so the
record names them; the code is identified by a hash of its sources in any case.

Recipe values can be overridden per run, as one shell-quoted list:
`env_variables.RFT_SET="train.total_training_steps=3 train.seed=7"`. Use the same overrides for
every stage of the run. `train` resumes from the run's newest published checkpoint: to continue
a finished run, submit `train` again with a larger `train.total_training_steps` (miles:
`train.num_rollout`).

To run every stage of a recipe in order from your workstation, filled in from `site.env`, with the
commit and image for the record, and the baseline evaluation beside training:

```bash
python -m rft_cookbook.air.chain officeqa/verl/qwen3.6-35b-a3b_h100x16 oqa-q36-01 --set "train.seed=7"
python -m rft_cookbook.air.chain math/verl/qwen3.5-2b_h100x8 math-smoke-01 --smoke   # GPU smoke (tier 5)
```

It keeps the jobs it submitted in `~/.rft-runs/<run id>.json`; run the same command again to resume
a chain: a job that succeeded is reused and one that still runs is waited for.

## Testing

```bash
pip install --no-deps -e . && pip install pytest pyyaml regex sympy pylatexenc openai
pytest tests                          # core: no RL framework needed
pytest frameworks/verl/tests          # needs the pinned verl importable (CPU is enough)
pip install --no-deps -e frameworks/miles && pip install transformers openai
pytest frameworks/miles/tests         # no miles needed; with torch and the pinned miles on
                                      # PYTHONPATH, also against miles itself
```

The verl tests run the cookbook agent loop inside verl's real `ToolAgentLoop` with a real Qwen3.5
tokenizer and chat template (only generation is scripted), compose every resolved training
argument against verl's own Hydra config, keep every checked-in job consistent with its recipe,
and check that the native loop and the Chat Completions harness make the same decisions on the
same episode. The miles tests check the arguments each recipe passes to miles' launcher, run the
agent function against a scripted session endpoint, and cover the reward, post-save and attestation
hooks and the checked-in jobs; against the pinned miles (`test_miles_api.py`) they load every hook
with miles' own loader, check it against the call miles makes, run miles' session post-processing
and check every flag a recipe passes against miles' parser. The core tests include the judge client
against a fake OpenAI-compatible server. [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs
them all on CPU, the miles tests twice: without miles and against the pinned miles.

## How to get help

Databricks support doesn't cover this content. For questions or bugs, please open a GitHub issue
and the team will help on a best effort basis.

## License

&copy; 2025 Databricks, Inc. All rights reserved. The source in this repository is provided subject
to the Databricks License [https://databricks.com/db-license-source]. Files derived from Apache-2.0
projects keep the Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE)); they are listed, with
every other third-party source, in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). All included
or referenced third-party libraries are subject to their own licenses.
