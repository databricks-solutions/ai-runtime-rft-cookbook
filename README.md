# AI Runtime RFT Cookbook

Train and evaluate tool-using LLM agents with reinforcement learning on Databricks AI Runtime.

The cookbook includes math, multi-hop search, and financial-document research tasks. Each recipe
prepares the data, evaluates a base model, trains it with GRPO, and evaluates the trained
checkpoint on the same held-out questions. Tasks share their tools and scoring across
[verl](https://github.com/verl-project/verl) and [miles](https://github.com/radixark/miles).

**Start with [math on verl](#quickstart).** It is the smallest end-to-end example: one node of
8 H100s and about 50 minutes.

## Tasks

| Task | What the agent does | Scoring |
|---|---|---|
| Math | Solves MATH problems with a calculator | Checks the final `\boxed{}` answer |
| Agentic search | Answers MuSiQue multi-hop questions using a Databricks Vector Search index | Exact match against the answer and its aliases |
| OfficeQA | Searches U.S. Treasury Bulletins, computes with retrieved figures, and cites its evidence | Checks the answer, then uses an LLM judge to assess the evidence |

## Tested recipes

These recipes have complete before-and-after evaluations. Training and rollout serving use
separate GPUs: a 16-H100 recipe uses 8 for training and 8 for rollouts.

| Recipe | Framework | Hardware | Held-out score, before → after | Change, 95% CI | Evidence |
|---|---|---|---|---|---|
| [Agentic search · Qwen3.6-35B-A3B](recipes/search/verl/qwen3.6-35b-a3b_h100x16/recipe.yaml) | verl | 16 H100 | Exact match **28.2% → 37.4%** | +9.2 points (+5.2 to +13.4) | [Results](docs/records/search-q36-01/qualification.json) |
| [OfficeQA · Qwen3.6-35B-A3B](recipes/officeqa/verl/qwen3.6-35b-a3b_h100x16/recipe.yaml) | verl | 16 H100 | Supported pass@1 **20.3% → 23.8%** | +3.5 points (−2.9 to +9.9) | [Results](docs/records/oqa-q36-verl-03/qualification.json) |
| [OfficeQA · Qwen3.6-35B-A3B](recipes/officeqa/miles/qwen3.6-35b-a3b_h100x16/recipe.yaml) | miles | 16 H100 | Supported pass@1 **20.3% → 23.8%** | +3.5 points (−3.5 to +11.6) | [Results](docs/records/oqa-q36-miles-03/qualification.json) |
| [Math · Qwen3.5-2B](recipes/math/verl/qwen3.5-2b_h100x8/recipe.yaml) | verl | 8 H100 | MATH-500 accuracy 61.8% → 62.0% | +0.2 points (−2.8 to +3.2) | [Results](docs/records/math-2b-02/qualification.json) |

The math recipe trains for two updates to test the workflow, not to demonstrate a quality gain.
OfficeQA's *supported pass@1* requires both a correct answer and evidence accepted by the judge.
Its confidence intervals include zero, so these runs do not establish a reliable improvement.
The intervals resample questions, keeping each question's evaluation samples together.

Search improves mainly by answering more questions rather than reaching the turn limit. It also
makes fewer tool calls, but a smaller share of its answers are correct. The search corpus
includes the held-out questions' source passages; this is not an evaluation against unseen
documents. See [results and evaluation details](docs/records/README.md).

### Other configurations

| Recipe | Hardware | Coverage and limitations |
|---|---|---|
| [Math · GLM-5.3-Flash, 4-layer slice · miles](recipes/math/miles/glm-5.3-flash-4layer_b300x8/recipe.yaml) | 8 B300 | Training and checkpoint publication tested. The slice is for integration testing, not model quality. |
| [Math · full GLM-5.3-Flash · miles](recipes/math/miles/glm-5.3-flash_b300x32/recipe.yaml) | 32 B300 | Training and resumable checkpoint publication tested; no held-out quality result. |
| [OfficeQA · full GLM-5.3-Flash · miles](recipes/officeqa/miles/glm-5.3-flash_b300x64/recipe.yaml) | 64 B300 | Experimental: fails during the first training update. |
| [OfficeQA · Qwen3.5-35B-A3B · verl](recipes/officeqa/verl/qwen3.5-35b-a3b_h100x16/recipe.yaml) | 16 H100 | Historical recipe, superseded by Qwen3.6. |

**Known miles limitation:** AI Runtime can report a training job as failed during shutdown even
after its checkpoint was published. The recent smoke test verifies the checkpoint and evaluation,
not clean job termination. See the [miles limitations](frameworks/miles/README.md#known-limitations).

## Requirements

- A Databricks workspace with AI Runtime and the GPUs required by your recipe.
- The [Databricks CLI](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli),
  authenticated to that workspace.
- A Unity Catalog volume for data, models, checkpoints, and evaluations. A 35B training checkpoint
  is about 0.5 TB; the recipes retain the newest two.
- A Hugging Face token stored as a Databricks secret.
- The [verl image](frameworks/verl/README.md#building-the-image), and the
  [miles image](frameworks/miles/README.md#building-the-image) if training with miles. Both
  frameworks use the verl image for evaluation.
- For agentic search, an existing Vector Search index with the
  [expected corpus and schema](src/rft_cookbook/tasks/search/__init__.py).

## Quickstart

1. Build the verl image and upload it to your workspace's Artifact Registry:

   ```bash
   bash frameworks/verl/image/build.sh rft-verl:<tag>
   databricks air images push -p <profile> --source rft-verl:<tag> --catalog <catalog> --schema <schema>
   ```

2. Copy `site.example.env` to `site.env`. Set your CLI profile, volume root, image name
   (`<catalog>.<schema>.rft-verl:<tag>`), image digest, and Hugging Face secret reference.
   Keep `site.env` private; it is gitignored.

3. Install the cookbook and run the math recipe:

   ```bash
   pip install -e .
   python -m rft_cookbook.air.chain math/verl/qwen3.5-2b_h100x8 math-01
   ```

   `math-01` is the run name you choose. If the command is interrupted, run it again with the
   same name and settings. It reuses successful stages and waits for jobs still running.

4. Read `runs/math-01/comparison.json` and `runs/math-01/qualification.json` on your volume.
   The comparison contains the scores; qualification records the configuration and checks.

For overrides, individual stages, resume, and troubleshooting, see
[Running recipes](docs/running-recipes.md).

## Development and testing

The framework-neutral tests run without GPUs or an RL framework:

```bash
pip install -e '.[dev,eval]'
pytest tests
```

Adapter tests need their own dependencies. See the
[verl](frameworks/verl/README.md#testing) and [miles](frameworks/miles/README.md#testing) guides;
[the CI workflow](.github/workflows/ci.yml) lists the pinned test environments.

For a short GPU test, `--smoke` trains two updates, restarts for a third, then exports and
evaluates the checkpoint:

```bash
python -m rft_cookbook.air.chain math/verl/qwen3.5-2b_h100x8 math-smoke-01 --smoke
```

The [architecture guide](docs/architecture.md) explains the shared agent policy, framework
adapters, checkpoint handling, and how to add a task.

## Repository layout

| Directory | Contents |
|---|---|
| `src/rft_cookbook/` | Tasks, agent policy, tools, scoring, evaluation, and checkpoint storage |
| `frameworks/` | Framework adapters, launchers, images, and adapter tests |
| `recipes/` | Recipe settings and the AI Runtime job for each stage |
| `docs/` | Usage, architecture, and published evaluation evidence |
| `tests/` | Framework-neutral tests |

## Help

For questions or bugs, open a GitHub issue. This cookbook is supported on a best-effort basis
and is not covered by Databricks support.

## License

&copy; 2025 Databricks, Inc. All rights reserved. The source in this repository is provided subject
to the Databricks License [https://databricks.com/db-license-source]. Files derived from Apache-2.0
projects keep the Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE)); they are listed, with
every other third-party source, in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). All included
or referenced third-party libraries are subject to their own licenses.
