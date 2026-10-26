# Architecture

This guide is for contributors adding a task, adapting a framework, or examining how an
experiment is evaluated. For setup and commands, see [Running recipes](running-recipes.md).

## Overview

A task defines the prompt, tools, budgets, stopping rule, and scorer. A framework adapter runs
that task and connects its reward to training. Evaluation uses the same task against vLLM.

```text
                           AgentTask
                               |
                         EpisodePolicy
                    /          |           \
             verl loop    miles agent    evaluation
                          (session v2)   (vLLM + chat)
                    \          |           /
                         EpisodeRecord
                               |
                        scorer and evidence
```

`EpisodePolicy` owns the decisions shared by all three drivers: which tool calls run, whether
observations fit, and when an answer ends an episode. Frameworks own their tokens, loss masks,
and log-probabilities. The adapters record enough information to link a scored episode to the
sample used for training.

Task code does not import an RL framework. Framework-specific code lives under `frameworks/`.
Adapters use upstream extension points; image patches are kept small, pinned, and documented
with a condition for removal.

## Tasks

A task is a package under `rft_cookbook.tasks`. Its `task()` function returns an `AgentTask`:

| Field | Purpose |
|---|---|
| `system_prompt` | Instructions shared by training and evaluation |
| `tools` | Executable tools and their OpenAI function schemas |
| `terminal` | A final-answer tool, or an answer pattern in a turn without tool calls |
| `budgets` | Turn, context, tool-call, output, and timeout limits |
| `scorer` | Answer checking, with an optional evidence judge |

Data is stored as JSONL rows with `id`, `split`, `messages`, `ground_truth`, and `metadata`.
Rows do not contain a system prompt; the task supplies it. Framework adapters convert these
rows to their own input format. Evaluation reads them directly.

### Adding a task

Use the [math task](../src/rft_cookbook/tasks/math/__init__.py) as the smallest example.

1. Implement `task(**budget_overrides)` and return an `AgentTask`.
2. Add `data.py` with `prepare_recipe(recipe, out, **_)`. Write canonical rows and a manifest
   that identifies their source and split.
3. If the task needs external resources, implement `run_resources(recipe, paths)` and
   `preflight(resources)`. OfficeQA uses a corpus, local tool service, and judge; search uses a
   Vector Search index.
4. Add a recipe extending `agentic/verl` or `agentic/miles`, with a checked-in job for each stage.
5. Test the tools, scorer, and stopping behavior with small recorded episodes before a GPU run.

## Episode format

`EpisodeRecord` is the common, versioned record for a training or evaluation attempt. It contains:

- task, question, model, sample, and configuration identities;
- model turns, visible text, reasoning, parsed tool calls, and token counts;
- tool outcomes, timeouts, truncation, sandbox level, and the observations delivered to the model;
- the stopping reason and final answer;
- the score, its validity, components, and any judge verdict;
- training attestations linking the episode to collected tokens and loss masks.

A record distinguishes an incorrect answer from an infrastructure or driver failure. That
separation is important: a failed judge or unreachable tool must not become a zero reward.

### Episode decisions

The drivers pass parsed calls and token counts to `EpisodePolicy`; the policy does not tokenize.

- Calls are considered in order, up to the per-turn limit. Extra calls are recorded as skipped.
- The first well-formed terminal call within that limit ends the episode. Other calls in the
  same turn are skipped because the model cannot receive their results.
- Malformed calls produce tool errors and consume the applicable call budget.
- A length-limited or empty turn ends without an answer.
- A turn that still needs tools at the turn limit ends with `max_turns`.
- A turn's observations enter the context together, only if they leave room for another turn.

The response budget includes generated text, tool observations, and template tokens appended
after the initial prompt. The context budget includes the initial prompt as well.

### Which samples train

| Episode or sample | Treatment |
|---|---|
| Answered, with the complete final answer retained | Use the scorer's reward |
| Ended without an answer | Use a valid zero when the scorer permits it |
| Collection dropped or truncated the scored answer | Exclude the sample |
| Invalid score, infrastructure failure, or driver failure | Exclude the sample |

Both frameworks drop an excluded sample's whole GRPO group. Repeated exclusions stop training
according to `train.max_consecutive_exclusions`. A judge outage first pauses scoring for up to
`train.judge_outage_wait_s`.

## Tools and sandboxing

A tool implements `call(arguments, context) -> ToolResult`. `ToolSet` applies a timeout and output
cap. Ordinary tool errors go back to the model; a backing-service failure raises `ToolInfraError`
and invalidates the episode.

OfficeQA hosts its corpus tools in one node-local HTTP service per node. This avoids loading
the corpus for every episode. The launcher starts the service before training and stops it
afterward.

Model-written Python runs in a separate interpreter:

| Level | Isolation |
|---|---|
| `strict` | bubblewrap namespaces, no network, a read-only root, private temporary storage, and masked data/credential directories. Unavailable isolation fails the call. |
| `best_effort` | Uses strict isolation when available; otherwise uses a resource-limited subprocess without a filesystem or network boundary. |
| `off` | Plain subprocess, for tests only. |

OfficeQA recipes select `strict`. Every compute call records the isolation it actually received.

## Scoring and judges

A scorer returns `RewardResult(valid, score, components)`. Judge policies identify the allowed
models, rubric, request budget, timeouts, retries, and fallback behavior. Calibration checks a
judge on known examples before it grades evaluation answers.

OfficeQA first checks answer correctness and the cited observation references. Only a correct
answer with a valid evidence path reaches the support judge. An unavailable or undecided judge
makes the score invalid, not incorrect. Evidence that cannot fit intact in the judge's request
budget is also left undecided.

## Framework adapters

| Adapter | Episode driver | Training layout | Checkpoint integration |
|---|---|---|---|
| [verl](../frameworks/verl/README.md) | Native `ToolAgentLoop` subclass | V1 `separate_async`, with separate trainer and vLLM pools | Post-save callback |
| [miles](../frameworks/miles/README.md) | Chat harness through session server v2 | Model-specific launcher; Qwen uses separate trainer and SGLang pools | Megatron post-save hook |

All exported models are evaluated through the same vLLM harness. Cross-framework comparisons
therefore share the evaluation implementation, not the training rollout engine.

## Checkpoints

Each adapter describes a resumable checkpoint as a `CheckpointBundle`: its model, optimizer,
random state, data position, and any rollout state, together with native completion markers.

`CheckpointStore` publishes the bundle to the volume and writes its completion marker last.
`latest()` ignores incomplete publications. Large recipes save directly to the volume; smaller
ones save locally and copy the completed bundle.

| | verl | miles |
|---|---|---|
| Directory | `global_step_N/` | `iter_N/` |
| Save interval | `train.save_freq` | `train.save_interval` |
| Resume option | `trainer.resume_mode=resume_path` | `--load` |

The export stage checks the Hugging Face weights. Retention keeps the newest `checkpoint.keep`
bundles and protects the newest bundle with complete exportable weights. A 35B checkpoint is
about 0.5 TB, including optimizer state. Those recipes avoid hashing the entire checkpoint at
every save; export hashes the final model's weight files instead.

## Recipes

Recipes extend a generic framework recipe. Mappings merge recursively; child lists and scalar
values replace their parents. Command-line overrides apply last. The resolved configuration is
hashed and included in the run's evidence.

Jobs are checked in beside each recipe as `jobs/<stage>.yaml`. Inside a job, the launcher
resolves that recipe from the uploaded snapshot and builds the framework arguments. Stages
exchange data through the run's volume directory. The [running guide](running-recipes.md)
describes the stages and how to resume them.

The packages are installed from the snapshot on every node before workers start. The framework
itself comes from its image, or from the pinned miles source installed at startup.

## Run records

The comparison stage writes `qualification.json`, identifying the recipe, model, data, code,
images, evaluation settings, hardware, and checks. `comparison.json` contains paired results and
confidence intervals. See [Results and test coverage](records/README.md) for published examples.

A `measured` record describes the full recipe, allowing changes to seeds, baseline source,
stages, and diagnostic logging. Other modifications, including a shorter smoke-test budget,
produce `smoke-tested` when the checks pass. Neither label is a general certification: read the
record's configuration, checks, and relevant framework limitations.

Comparisons must use the same base-model identity, questions, agent rules, scorer, and evaluation
settings. Confidence intervals resample questions, not individual completions of the same
question. Training probes are reported separately from held-out questions.

Publishing with `rft_cookbook.evidence.publish` replaces workspace details with placeholders and
records the originals' hashes. Historical records retain the checks and hash formats that were
available when they were produced.

## Testing

| Test group | Purpose | Environment |
|---|---|---|
| Unit | Agent policy, tools, scorers, episode schema, and evidence accounting | CPU |
| Adapter contract | Reward eligibility, token retention, and sample conversion | CPU |
| Framework API | Hooks and configuration against pinned upstream code | CPU with framework source |
| Recipe and packaging | Job consistency, package imports, and public-data checks | CPU |
| GPU smoke | Short training, restart/resume, export, and evaluation | AI Runtime |
| Full recipe | The declared training budget and held-out evaluation | AI Runtime |

[The CI workflow](../.github/workflows/ci.yml) defines the CPU environments and commands. GPU
runs are maintainer-triggered, not executed on untrusted pull-request code. CPU tests do not
validate GPU kernels or replace testing in the actual framework images.

## Public repository rules

Keep workspace settings in the gitignored `site.env`. Store secret references there, never
secret values in source. Published evidence must be sanitized before committing it.

Datasets and corpora are downloaded during preparation rather than redistributed. Third-party
source retains its original license and attribution; see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
