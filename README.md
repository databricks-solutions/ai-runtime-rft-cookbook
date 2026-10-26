# AI Runtime RFT Cookbook

Recipes for training tool-using LLM agents with reinforcement learning on Databricks AI Runtime.

The agents in this cookbook work over many turns. They call tools (a calculator, a document
search, a vector index), read the results and decide what to do next, until they give an answer.
A full recipe takes an open model from Hugging Face and evaluates it on held-out questions. It
then trains the model with GRPO and evaluates it again, so you can see what training changed.

A task (its tools, prompt and scoring) is written once and runs on more than one RL framework:
[verl](https://github.com/verl-project/verl) and [miles](https://github.com/radixark/miles)
today, slime next. Each framework is connected through its own documented extension points, with
no fork.

## Tasks

| Task | What the agent does | How an answer is scored |
|---|---|---|
| **Math** | Solves competition math problems (MATH), using a calculator for arithmetic | Rule-based check of the final `\boxed{}` answer |
| **Agentic search** | Answers multi-hop questions (MuSiQue) by searching a Databricks Vector Search index of encyclopedia passages | Exact match against the reference answer and its aliases |
| **OfficeQA** | Answers questions about U.S. Treasury Bulletins by searching and reading the documents and computing, then cites the evidence for each number | Correct answer, and an LLM judge (GLM-5.3-Flash on Databricks) must accept the cited evidence |

## Results

All the measured recipes train on separate GPUs from the ones serving rollouts. For example,
"16 H100" means 8 GPUs train and 8 serve.

| Task | Framework | Model | GPUs | Held-out score, before → after | Change (95% CI) | Time | Record |
|---|---|---|---|---|---|---|---|
| [Agentic search](recipes/search/verl/qwen3.6-35b-a3b_h100x16/recipe.yaml) | verl | Qwen3.6-35B-A3B | 16 H100 | Exact match **28.2% → 37.4%** | +9.2 points (+5.2 to +13.4) | 3 h | [record](docs/records/search-q36-01/qualification.json) |
| [OfficeQA](recipes/officeqa/verl/qwen3.6-35b-a3b_h100x16/recipe.yaml) | verl | Qwen3.6-35B-A3B | 16 H100 | Supported pass@1 **20.3% → 23.8%** | +3.5 points (−2.9 to +9.9) | 7.5 h | [record](docs/records/oqa-q36-verl-03/qualification.json) |
| [OfficeQA](recipes/officeqa/miles/qwen3.6-35b-a3b_h100x16/recipe.yaml) | miles | Qwen3.6-35B-A3B | 16 H100 | Supported pass@1 **20.3% → 23.8%** | +3.5 points (−3.5 to +11.6) | 10 h | [record](docs/records/oqa-q36-miles-03/qualification.json) |
| [Math](recipes/math/verl/qwen3.5-2b_h100x8/recipe.yaml) (quickstart) | verl | Qwen3.5-2B | 8 H100 | MATH-500 accuracy 61.8% → 62.0% | +0.2 points (−2.8 to +3.2) | 50 min | [record](docs/records/math-2b-02/qualification.json) |

The task names link to the recipes. The math recipe trains for only two updates: it checks that
every stage works, and isn't expected to improve the model.

*Supported pass@1* is the share of answers that are correct **and** whose cited evidence the
judge accepts. The confidence intervals are paired bootstrap intervals over questions. *Time* is the
wall-clock time of the whole recorded run, from data preparation to the final comparison.

What the results show:

- **Agentic search** improves the most, mainly because the trained model commits to an answer.
  It answers 87% of the questions, where the base model answers 48%; most of the base model's
  other episodes reach the 16-turn limit. The trained model also searches
  less (5.7 tool calls per question instead of 10.0). The cost is precision: 43% of its answers
  are right, against 58.5% for the base model. The index holds the passages of every held-out
  question, so each answer can be found.
- **OfficeQA** gains about three points in 100 updates. Three verl runs gained between 0 and 3.5
  points, so no single run is conclusive. verl and miles, trained with matched settings, reach the
  same score.
- **On miles, turn off the MoE load-balancing loss** to match verl. With it on, the miles OfficeQA run
  fell to 11.0%.

### Large-scale and older recipes

| Recipe | Framework | Model | GPUs | Status |
|---|---|---|---|---|
| [Math, smoke test](recipes/math/miles/glm-5.3-flash-4layer_b300x8/recipe.yaml) | miles | GLM-5.3-Flash, a 4-layer slice | 8 B300 | **Ran**: two updates and a published checkpoint |
| [Math, full model](recipes/math/miles/glm-5.3-flash_b300x32/recipe.yaml) | miles | GLM-5.3-Flash (about 300B) | 32 B300 | **Ran**: three updates and a resumable 4.8 TB checkpoint |
| [OfficeQA, full model](recipes/officeqa/miles/glm-5.3-flash_b300x64/recipe.yaml) | miles | GLM-5.3-Flash (about 300B) | 64 B300 | **Experimental**: both attempts stopped in the first update (see the recipe) |
| [OfficeQA, Qwen3.5](recipes/officeqa/verl/qwen3.5-35b-a3b_h100x16/recipe.yaml) | verl | Qwen3.5-35B-A3B | 16 H100 | **Superseded** by the Qwen3.6 recipe. Two seeds: 15.1% → 22.1% and → 20.9% |

What the statuses mean:

- **Measured**: a full run of the recipe as committed passed every check of its run record. The
  checks are: the evaluation is complete, training and test questions don't overlap, the
  evaluated model is the trained checkpoint, and every stage ran the same code. Records are in
  [`docs/records`](docs/records/README.md).
- **Ran**: the recipe's GPU runs passed before run records existed. Its header describes them.
- **Experimental**: the recipe doesn't complete yet.
- **Superseded**: kept with its results; a newer recipe replaces it.

## What you need

- A Databricks workspace with AI Runtime, and GPUs: H100 nodes for the Qwen recipes, B300 nodes
  for GLM-5.3-Flash.
- The [Databricks CLI](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli), whose
  `databricks air` commands submit and watch the jobs, and a CLI profile for the workspace.
- A Unity Catalog volume for data, models, checkpoints and evaluations. Plan on a lot of space for
  the 35B recipes: each checkpoint is about 0.5 TB, and they keep the newest two.
- A Hugging Face token, stored as a Databricks secret.
- The framework's container image, built from `frameworks/<framework>/image` and pushed into your
  workspace's Artifact Registry with `databricks air images push`. See
  [frameworks/verl](frameworks/verl/README.md) and [frameworks/miles](frameworks/miles/README.md). miles recipes need the verl image too: every
  recipe evaluates with vLLM from the verl image, so results compare across frameworks.
- For agentic search, a Vector Search index of the corpus. The cookbook doesn't build it;
  [`rft_cookbook/tasks/search`](src/rft_cookbook/tasks/search/__init__.py) describes its contents
  and schema.

## Quickstart: math on verl

The math recipe is the smallest. It runs on one node of 8 H100s in about 50 minutes and exercises
every stage: data preparation, evaluation, training, checkpointing, export and comparison.

1. Build the verl image and push it into your workspace
   ([details](frameworks/verl/README.md#building-the-image)):

   ```bash
   bash frameworks/verl/image/build.sh rft-verl:<tag>
   databricks air images push -p <profile> --source rft-verl:<tag> --catalog <catalog> --schema <schema>
   ```

2. Copy `site.example.env` to `site.env` and fill in your workspace profile, volume root, image
   (`<catalog>.<schema>.rft-verl:<tag>`) and Hugging Face secret. `site.env` is gitignored.

3. Install the cookbook on your workstation and run the recipe:

   ```bash
   pip install -e .
   python -m rft_cookbook.air.chain math/verl/qwen3.5-2b_h100x8 math-01
   ```

   `math-01` is your name for the run. The command submits each stage as an AI Runtime job, waits
   for it, and then submits the next one. If it is interrupted, run the same command again: stages
   that already succeeded are skipped, and running ones are waited for.

4. When it finishes, the run's results are on your volume under `runs/math-01/`.
   `comparison.json` has the before/after scores and `qualification.json` is the run record.

## Running a recipe

A recipe is a YAML file that pins one task to one framework, model and GPU layout. Every recipe
runs as a sequence of stages, each a separate AI Runtime job:

| Stage | What it does |
|---|---|
| `prep` | Downloads the task's data and prepares it for the framework |
| `stage_model` | Copies the model, at its pinned revision, to the volume |
| `stage_judge`, `calibrate` | OfficeQA only: prepares the judge and checks it on known cases before it grades anything |
| `baseline_eval` | Evaluates the base model on the held-out questions (runs beside training) |
| `train` | Trains with GRPO, publishing checkpoints to the volume |
| `export` | Checks the Hugging Face weights of the final checkpoint |
| `eval` | Evaluates the trained model the same way as the base model |
| `compare` | Compares the two evaluations question by question and writes the run record |

The chain runner from the quickstart is the easiest way to run them. It also takes overrides,
which apply to every stage:

```bash
python -m rft_cookbook.air.chain officeqa/verl/qwen3.6-35b-a3b_h100x16 oqa-01 --set "train.seed=7"
```

You can also submit a stage yourself. Each stage's job is checked in next to its recipe
(`recipes/<task>/<framework>/<recipe>/jobs/<stage>.yaml`), with placeholders for your workspace
and run:

```bash
source site.env
databricks air run -f recipes/math/verl/qwen3.5-2b_h100x8/jobs/prep.yaml -p "$RFT_PROFILE" \
  --override env_variables.RFT_RUN_ID=math-01 --override env_variables.RFT_VOLUME_ROOT="$RFT_VOLUME_ROOT" \
  --override environment.unity_catalog_image="$RFT_VERL_IMAGE" --override secrets.HF_TOKEN="$RFT_HF_TOKEN_SECRET"
```

When you submit stages yourself:

- Use the same run id and the same overrides for every stage of a run (overrides go in
  `env_variables.RFT_SET`, for example `RFT_SET="train.seed=7"`).
- Agentic search also needs `env_variables.RFT_VS_INDEX` in `baseline_eval`, `train` and `eval`.
- Pass `env_variables.RFT_CODE_COMMIT` and `env_variables.RFT_IMAGE` so that the run record names
  the commit and image. The chain runner does this for you.

To continue a finished run, submit `train` again with more steps
(`train.total_training_steps` on verl, `train.num_rollout` on miles). It resumes from the run's
newest checkpoint.

## Repository layout

```
src/rft_cookbook/    framework-neutral code: the agent and its episode format, tools, scoring
                     and judges, the tasks, checkpoints, evaluation and run records
frameworks/verl/     the verl adapter, its job launcher and its image
frameworks/miles/    the miles adapter, its job launcher, model launchers and its image
recipes/             the recipes, and one generic agentic recipe per framework that they extend
docs/                the design, and the published run records
tests/               CPU tests of the framework-neutral code
```

A verl user installs and imports nothing from miles. To learn how the
pieces fit together, or to add a task or a framework, read
[docs/architecture.md](docs/architecture.md).

## Testing

The CPU tests need no GPU and no RL framework:

```bash
pip install --no-deps -e . && pip install pytest pyyaml regex sympy pylatexenc openai
pytest tests                          # framework-neutral code, every recipe and job, public-safety scan
pytest frameworks/verl/tests          # the verl adapter; needs the pinned verl installed (CPU is enough)
pip install --no-deps -e frameworks/miles && pip install transformers openai
pytest frameworks/miles/tests         # the miles adapter; with the pinned miles installed, also against it
```

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) lists the exact installs. The verl tests
run the cookbook's agent loop inside verl's real `ToolAgentLoop`, with a real tokenizer and chat
template. With the pinned miles installed, the miles tests load every hook through miles' own loader and
check every flag against miles' parser.

GPU tests run on AI Runtime. `--smoke` runs a short version of a recipe: two updates, a restart
and resume, export and evaluation:

```bash
python -m rft_cookbook.air.chain math/verl/qwen3.5-2b_h100x8 math-smoke-01 --smoke
```

The latest result of every test tier is in [docs/records](docs/records/README.md).

## How to get help

Databricks support doesn't cover this content. For questions or bugs, please open a GitHub issue
and the team will help on a best effort basis.

## License

&copy; 2025 Databricks, Inc. All rights reserved. The source in this repository is provided subject
to the Databricks License [https://databricks.com/db-license-source]. Files derived from Apache-2.0
projects keep the Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE)); they are listed, with
every other third-party source, in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). All included
or referenced third-party libraries are subject to their own licenses.
