# Architecture

This page explains how the cookbook is put together: what a task is, how an episode runs on each
framework, and how checkpoints and run records work. Read it if you want to add a task or a
framework, or to understand exactly what a result means. To run a recipe, the
[README](../README.md) is enough.

## Overview

```
                    AgentTask  (prompt, tools, when to stop, budgets, scorer, data)
                        │
                   agent.policy      the shared rules: which tool calls run,
                        │            when an episode ends, what enters the context
        ┌───────────────┼─────────────────────────┐
        │               │                         │
  verl agent loop   Chat Completions        Chat Completions
  (rft_verl)        harness behind          harness against
                    miles' session          a vLLM server
                    server (rft_miles)      (evaluation)
        │               │                         │
        └───────────────┼─────────────────────────┘
                        │
                  EpisodeRecord  ──►  scorer  ──►  reward, metrics, run record
```

- A **task** describes the job: the system prompt, the tools, how an episode ends with an
  answer, the budgets, and how an answer is scored.
- An **episode** is one attempt at one question. The rules of an episode live in one shared
  module, `rft_cookbook.agent.policy`.
- Each framework **drives** episodes its own way. verl uses its native agent loop, miles uses a
  Chat Completions harness behind its session server, and evaluation uses the same harness
  against a vLLM server. All three call the same rules.
- Every episode, in training or evaluation, produces the same **`EpisodeRecord`**. Scoring,
  metrics and run records read only that record, so they work the same for every framework.

## Design principles

1. **Tasks don't depend on frameworks.** A task's tools, prompt and scorer import no RL framework.
   Only the adapters under `frameworks/` do.
2. **Shared rules, native drivers.** Each framework runs episodes in its own way, but every one
   makes the same decisions by calling `agent.policy`.
3. **The framework owns the training tokens.** Sampled token ids, loss masks and log-probabilities
   stay in the framework. The adapter records that the episode that was scored is the one that
   trained.
4. **No forks.** Adapters use only documented extension points of the pinned framework version. A
   patch to a framework's image is a last resort, with a written reason and a condition for
   removing it.
5. **Frameworks save, the cookbook publishes.** Each framework writes checkpoints in its native
   format. The cookbook copies them to the volume, verifies them, resumes from them and exports
   them, behind one interface.
6. **Evaluation doesn't depend on the framework.** Every trained checkpoint is exported to Hugging
   Face weights and evaluated by the same harness on vLLM, so results compare across frameworks.
7. **Nothing workspace-specific is committed.** Checked-in jobs leave placeholders that you fill
   in from your own `site.env` when you submit them.

## Tasks

A task is a Python package, `rft_cookbook.tasks.<name>`, whose `task()` function returns an
`AgentTask`:

| Part | What it is | OfficeQA, for example |
|---|---|---|
| system prompt | The instructions; the run records its hash | A research agent's instructions |
| tools | Tool implementations and their OpenAI JSON schemas | search, grep, read, list, compute |
| terminal rule | How an episode ends with an answer: a final tool call, or an answer pattern in a turn without tool calls | the `submit_report` tool |
| budgets | Turns, tool calls per turn, context tokens, tokens per turn, tool output size, timeouts | 128 turns, 4,096 tokens a turn, a 98,304-token context, 8,000-byte tool results |
| scorer | Rule-based, or with an LLM judge | answer check, then a judge of the cited evidence |

Its data module prepares the rows every framework reads: `id`, `split`, `messages`,
`ground_truth` and `metadata`. A recipe can change a task's budgets, scorer settings and data
selection, but not its tool code.

### Adding a task

The math task ([`src/rft_cookbook/tasks/math`](../src/rft_cookbook/tasks/math/__init__.py)) is
the smallest example to copy.

1. Create `src/rft_cookbook/tasks/<name>/__init__.py` with a `task(**budget_overrides)` function
   that returns an `AgentTask`: a name and version, the system prompt, a `ToolSet`, a
   `TerminalRule`, `Budgets` and a scorer.
2. Add `data.py` with `prepare_recipe(recipe, out, **_)`. It writes `train.jsonl` and `eval.jsonl`
   with `rft_cookbook.data.write_rows`, and a manifest of the sources.
3. If the task needs resources outside the code, add `run_resources(recipe, paths)` to name them
   and `preflight(resources)` to check them before a stage starts. For example, search names its
   Vector Search index and checks that it is ready. OfficeQA names its tool service, corpus and
   judge.
4. Write a recipe that extends `agentic/verl` or `agentic/miles`, and copy the `jobs/` folder of
   a similar recipe next to it. `pytest tests` checks that the jobs match the recipe.
5. Add CPU tests with small recorded episodes.

## Episode format

`EpisodeRecord` is versioned JSON, written for every training and evaluation episode. It holds:

- **identity**: the episode, task and task version, question, split and sample;
- **policy**: the model and its revision or checkpoint, and the policy version behind each turn;
- **agent settings**: hashes of the prompt, tool schemas, chat template, tool-call parser and
  budgets, so that runs with different settings never compare by accident;
- **turns**: for the model, its text, reasoning, parsed tool calls (with parse errors), finish
  reason and token counts. For each tool call, its status (`ok`, `error`, `timeout`, `rejected` or
  `skipped`), whether it ran, whether its output reached the model, truncation, latency and the
  sandbox level it ran in;
- **termination**: why the episode ended (`answer`, `no_answer`, `max_turns`, `context_budget`,
  `tool_budget`, `length`, `policy_error` or `infra_error`) and the final answer;
- **score**: the value and its components, whether it is valid, the scorer's policy and the
  judge's verdict;
- **status**: `ok`, `policy_failure` or `infra_failure`;
- **collection** (training only): one **attestation** per training sample made from the episode.
  It records the sample's token and loss-token counts, a hash of its tokens and loss mask, and
  whether the whole final answer survived into the sample. The tokens themselves stay in the
  framework.

### The rules of an episode

`agent.policy` makes these decisions for every driver. It works on the driver's parsed tool calls
and token counts; it doesn't tokenize.

- Tool calls in one turn run in order, up to the per-turn limit. Calls past the limit are
  recorded as `skipped`, including a final-answer call.
- The first well-formed final-answer call within the limit ends the episode. The turn's other
  calls are `skipped`, since the model would never see their results.
- A malformed call, including a malformed final-answer call, goes back to the model as a tool
  error and counts toward the limit.
- A turn cut off by the token limit ends the episode without an answer, even if a final-answer
  call could be parsed from it. So does an empty turn.
- The budgets are separate: total context tokens, total generated tokens, a reserve of tokens the
  next turn needs, and the size of each tool result.
- One turn's tool results enter the context together, and only if they leave room for the next
  turn. Otherwise the episode ends with `context_budget`.
- Every driver separates a turn's `<think>` reasoning from its visible text the same way. The
  answer is read from the visible text.

### Which samples train

| Case | Reward |
|---|---|
| The episode answered, and the whole answer is in the training sample | The scorer's reward |
| The episode ended without an answer (turn limit, empty or cut-off turn) | A valid zero, if the scorer allows it |
| The training sample cut off or dropped the answer that was scored | None: the sample is excluded |
| The score is invalid (for example, the judge didn't answer), or the episode failed for infrastructure reasons | None: the sample is excluded |

An excluded sample never reaches the update. Both adapters drop its whole GRPO group, so it
doesn't bias the group's baseline either, and both count the exclusions. If too many episodes in
a row are excluded (`train.max_consecutive_exclusions`), the run stops instead of spinning. When
the judge is the cause, training first waits for it to come back, for up to
`train.judge_outage_wait_s`.

## Tools and sandboxing

A tool implements `call(arguments, context) -> ToolResult`, with a timeout and an output cap.
Light tools run in-process. Heavy ones (OfficeQA's corpus search) run in a **tool service** on
each node: an HTTP server that loads the corpus once and serves every episode on that node. The
launcher starts it before the framework and stops it after. If the service is unreachable, the
episode is an infrastructure failure, not a tool error.

Tools that run model-written code use a sandbox:

| Level | What it means |
|---|---|
| `strict` | bubblewrap: separate user, PID and network namespaces, a read-only root, an empty `/proc`, a private `/tmp`, and the workspace's data directories hidden. Fails if it isn't available. |
| `best_effort` (default) | `strict` where available, otherwise a resource-limited subprocess without guaranteed network isolation; the fallback is logged |
| `off` | A plain subprocess, for tests only |

Every tool call records the level it actually ran at.

## Scoring and judges

A scorer returns `RewardResult(valid, score, components)`. A judge scorer follows a versioned
**judge policy**: the rubric and what part of the episode the judge sees, the allowed judge
models, concurrency limits, timeouts and retries. A judge that doesn't return a verdict makes the
score invalid, never zero: a judge outage must not look like a wrong answer. Each judge model is
checked on the task's known cases (the `calibrate` stage) before it grades anything.

## Framework adapters

Each adapter connects a framework to the cookbook. It converts the data, registers the tools,
drives episodes, hands the reward to the framework, records attestations, publishes checkpoints
and builds the launch command. Adapters are the only code that imports a framework.

- **verl** ([frameworks/verl](../frameworks/verl/README.md)): a subclass of verl's
  `ToolAgentLoop` drives each episode and scores it with the task's scorer. verl's V1 trainer runs
  in `separate_async` mode, with training and a vLLM rollout pool on separate GPUs. A post-save
  callback publishes each checkpoint.
- **miles** ([frameworks/miles](../frameworks/miles/README.md)): upstream miles, unmodified. The
  cookbook's Chat Completions harness drives each episode against miles' session server, which
  records the tokens. Hooks passed as command-line flags provide the reward, the attestations and
  checkpoint publishing.

The framework READMEs list the extension points each adapter uses and their known limits.

## Checkpoints

A framework's "checkpoint saved" signal doesn't mean the checkpoint is complete or durable. verl
may still be writing, resuming needs more than the model weights, and miles saves its rollout
state separately. So each adapter describes a checkpoint as a **bundle**: the files it needs,
where each lives, and how to tell that each is complete.

`CheckpointStore` gives every framework the same operations on bundles:

- **publish**: copies the bundle to the volume, checks it against a manifest, and writes a
  completion marker last. Recipes with very large checkpoints save straight to the volume and are
  published in place, without a copy.
- **latest**: the newest bundle with a completion marker. Partial copies are ignored.
- **resume**: the framework's load arguments for that bundle.
- **export**: finds and checks the Hugging Face weights the framework wrote.
- **retention**: keeps the newest `checkpoint.keep` bundles, plus the newest one with complete
  Hugging Face weights.

| | verl | miles |
|---|---|---|
| A checkpoint | `global_step_N/` | Megatron `iter_N/`, with Hugging Face weights in `iter_N/hf` |
| Saved every | `train.save_freq` updates | `train.save_interval` updates |
| Resumed with | `trainer.resume_mode=resume_path` | `--load` |

A 35B checkpoint is about 0.5 TB (70 GB of weights, the rest optimizer state). Those recipes
check sizes instead of hashing every file (`hash_files: false`), and keep the newest two.

## Recipes

A recipe is YAML that extends the generic agentic recipe of its framework. Here is an abridged
version of the OfficeQA recipe on verl:

```yaml
extends: agentic/verl
task: officeqa
stages: [prep, stage_model, stage_judge, calibrate, baseline_eval, train, export, eval, compare]
model: {id: Qwen/Qwen3.6-35B-A3B, revision: <commit>, tool_parser: qwen3_coder}
scoring:
  sandbox: strict
  judge: {id: glm-5.3-flash-hosted-support-v1, provider: databricks, models: [system.ai.glm-5-3-flash]}
eval: {samples: 4, temperature: 1.0}
train: {rollout_n: 4, total_training_steps: 100, lr: 1.0e-6, save_freq: 25}
checkpoint: {location: volume, keep: 2}
compute: {nodes: 2, accelerator: GPU_8xH100, trainer_gpus: 8, rollout_gpus: 8}
```

How `extends` resolves: the parent's path is relative to `recipes/`. Mappings merge recursively,
a child's lists and scalars replace the parent's, and command-line overrides (`RFT_SET`) apply
last. Unknown keys and cycles are errors. The resolved recipe is hashed, and the hash goes into
the run record.

Each stage of a recipe is one AI Runtime job, checked in as `jobs/<stage>.yaml` next to the
recipe. Nothing is generated at submission. Inside the job, the launcher resolves the recipe from
the uploaded code and writes the framework's arguments. Stages pass their outputs to each other
through the run's directory on the volume. Training runs on the recipe's nodes, model staging and
evaluation on one GPU node, and the other stages on a small A10 node.

## Run records

The `compare` stage writes the run's record, `qualification.json`. It names everything the
result depends on: the code (a commit and a hash of the sources), the recipe, the image, the model
and tokenizer revisions, the data, the evaluation settings and the GPU layout. It also lists the
checks the run passed:

- **calibration** (judged tasks): the judge got every known case right;
- **export**: the final checkpoint's Hugging Face weights are complete, and unchanged since the
  `export` stage checked them (for recipes that don't hash every file, by file names and sizes);
- **evaluated checkpoint**: the `eval` stage served that checkpoint;
- **evaluations complete**: every planned evaluation episode is present, and each is scored or
  explicitly an infrastructure failure;
- **comparable**: the base and trained evaluations used the same questions and settings;
- **leakage clean**: training and evaluation questions don't overlap, by id or by content;
- **same code**: every stage ran the same code;
- **parity** (where a recipe has one): the base model's evaluation agrees with a recorded
  reference evaluation.

The comparison pairs the two evaluations question by question. Its confidence intervals come from
a bootstrap that resamples questions, because several samples of one question aren't independent.

A run is **measured** when it passed every check on the recipe as committed. Only the seeds, the
list of stages, and a baseline borrowed from another run (`eval.baseline_run`) may differ. A run
that passed every check with other changes, such as `--smoke`, is **smoke-tested**. A change to
the task, the rules of an episode, the scorer, an adapter, a launcher or an image needs a new run.

To publish a record, `python -m rft_cookbook.evidence.publish` replaces workspace paths and image
registries with placeholders, and records the hash of every original file. Published records are
in [`docs/records`](records/README.md).

A comparison across frameworks compares recipes: each run is reported against its own baseline,
unless the model, data, budgets, judge, evaluation and training budget all match. The two
Qwen3.6 OfficeQA recipes are matched this way, and the miles run is compared with the verl run's
baseline.

## Packaging

`rft-cookbook` and the adapters (`rft-verl`, `rft-miles`) are small Python packages. At the start
of a job they are installed from the uploaded code with `pip install --no-deps` on every node,
before Ray or any worker starts. Every process then imports the same code. They depend only on
what the framework images already contain, and they never declare a framework as a dependency:
frameworks come from their images.

## Testing

| Tier | What it covers | Where it runs |
|---|---|---|
| 1. Unit | The rules of an episode, tools, the harness against a fake chat server, judges against fake endpoints, sandbox detection, run-record checks, the episode schema | CPU |
| 2. Adapter contract | Each adapter turns recorded episodes into valid records, every adapter scores a recorded episode the same, attestations on truncated and retried episodes, invalid rewards never reach an update | CPU, with fake framework objects |
| 3. Framework API | The adapters against the real pinned framework: hook signatures, base classes, configuration | CPU, with the pinned framework installed |
| 4. Jobs and public safety | Every recipe resolves, its jobs match it, and no source contains a secret or workspace detail | CPU |
| 5. GPU smoke | A short run of a recipe: two updates, restart and resume, export, evaluation of the export (`--smoke`) | AI Runtime |
| 6. Full recipe | The full run, with its run record | AI Runtime |

The CPU tiers are meant to run on every pull request. The GPU tiers are started by a maintainer
from a reviewed commit, and never for pull requests from forks. The latest result of each tier is
in [`docs/records`](records/README.md).

## Public repository rules

- `site.env` holds your workspace profile, volume root, images and secret names (never secret
  values). It is gitignored and left out of the code uploaded to jobs.
- The tests scan sources for credentials, e-mail addresses, workspace hosts, user paths and run
  URLs.
- Published records contain aggregates and per-question scores, and list what was replaced during
  sanitization.
- Datasets and corpora are downloaded when a run prepares its data; they aren't redistributed.
- New code is under the repository license. Files derived from Apache-2.0 sources keep that
  license, with attribution; [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) lists them.

## Roadmap

- **slime**, as a third framework, then others on demand.
- **verl through a Chat Completions gateway**, compared with the native agent loop on a fixed
  checkpoint.
- **GLM-5.3-Flash on OfficeQA at 64 GPUs**, which stops in its first update today (see the recipe).
