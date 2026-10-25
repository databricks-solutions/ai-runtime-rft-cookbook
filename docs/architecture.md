# Architecture

This cookbook runs reinforcement fine-tuning (RFT) of tool-using agents on Databricks AI
Runtime with more than one RL framework. Agentic tasks are written once. Each framework
attaches through a thin adapter built on the framework's own extension points, with no
fork. A recipe pins one task to one framework, model and GPU layout.

Frameworks: **verl** (0.10.0.dev, pinned commit) and **miles** (upstream radixark/miles,
pinned commit and official image) first; **slime** later.

## Principles

1. **Task first.** Tools, prompts, scoring and evaluation of a task exist once and do not
   import any RL framework.
2. **One agentic recipe.** Every agentic task fills in the same specification (`AgentTask`).
   Each framework has one generic agentic recipe that runs any `AgentTask`. OfficeQA is the
   first task; agentic search and the math quickstart are the next instances. The
   abstraction grows only when a second task needs it.
3. **One episode format.** Every framework, and evaluation, produces the same `EpisodeRecord`.
   Scorers, metrics, evidence and tests read only that record.
4. **Shared decisions, native drivers.** The rules of an episode (which tool calls run,
   when the episode ends, what enters the context) are one shared module. Each framework
   may drive the episode its own way (verl's native agent loop, a Chat Completions harness
   behind miles' session server), but it calls the same decision code.
5. **Training tokens stay with the framework.** Exact sampled token IDs, loss masks and
   log-probabilities are owned by the framework; the adapter attests in the episode record
   that the scored episode is the episode retained for training.
6. **Framework-native checkpointing.** Frameworks save; the cookbook publishes, verifies,
   resumes and exports through one interface and handles each framework's layout.
7. **Isolation when available.** Code-executing tools run in the strongest sandbox the
   runtime allows; the level reached is recorded in every episode.
8. **No forks.** Adapters use only documented extension points of the pinned framework.
   A missing capability is implemented in the adapter or contributed upstream; an image
   patch is a last resort with a recorded reason and removal condition.
9. **Evidence is first-class and shared.** Completeness, leakage, identity and paired
   comparison checks are the same for every agentic task.
10. **Public by default.** Nothing workspace-specific is committed; a recipe's checked-in jobs
    leave placeholders that `air run --override` fills in from the user's `site.env`.

## Layers

```
          AgentTask  (prompt, tools, terminal rule, budgets, scorer, data)
              │
     agent.policy  ── shared decisions: call selection, termination, observation admission
       │                    │                         │
 verl native loop     Chat Completions harness     Chat Completions harness
 (rft_verl)           behind miles session server   against a serving endpoint
       │              (rft_miles)                   (evaluation)
       └──────────────┬───────┴─────────────────────────┘
                 EpisodeRecord  ──►  scorer  ──►  reward / metrics / evidence
```

## The agentic task specification

A task provides:

| Part | What | Example (OfficeQA) |
|---|---|---|
| data | canonical rows: `id`, `split`, `messages`, `ground_truth`, `metadata`; split manifests with content hashes | official split, held-out ids |
| prompt | system prompt file, versioned by hash | research-agent instructions |
| tools | tool implementations + OpenAI JSON schemas | search, grep, read, list, compute |
| terminal rule | how an episode ends with an answer: a terminal tool or an answer pattern | `submit_report` |
| budgets | max turns, max tool calls per turn, total context tokens, per-turn generation tokens, tool output bytes, wall time | 128 turns, 4,096 tokens a turn, 8,000 bytes per tool result |
| scorer | rule or judge scorer with a versioned policy | answer check + evidence judge |
| fixtures | small recorded episodes and a tiny corpus for CPU tests | — |

A task is a Python package, `rft_cookbook.tasks.<name>`, whose `task()` returns its `AgentTask`.
Recipes override budgets, scorer policy and data splits; they do not change tool code.

## Episode format

`EpisodeRecord` is versioned JSON, written for every training and evaluation episode:

- **identity:** episode id, task and task version, question id, split, sample index, attempt.
- **policy:** model id and revision or checkpoint id, policy version(s) that generated each turn.
- **agent settings:** hashes of prompt, tool schemas, chat template and template arguments,
  tool-call parser, budgets.
- **turns:** for the model: text, reasoning (if any), parsed tool calls with raw arguments and
  parse errors, finish reason, token counts. For tools: call id, observation id, status
  (`ok | error | timeout | rejected | skipped`), execution (ran or not) and delivery
  (`delivered | not_admitted | not_run`) recorded separately, the bytes delivered into the
  context and the model turn that first saw them, truncation, latency, sandbox level reached.
  An empty delivered output and an executed but unadmitted observation stay distinguishable.
- **termination:** reason (`answer | no_answer | max_turns | context_budget | tool_budget |
  length | policy_error | infra_error`), final answer, terminal turn.
- **score:** value, components, `valid`, scorer policy id, judge identity, verdict and attempts.
- **status:** `ok | policy_failure | infra_failure`, derived from the termination
  (`policy_error` is a driver that stopped without a decision). Neither failure is scored or
  converted into a learning signal; the scoring policy decides between retry and exclusion.
- **collection** (training only): a list of **attestations**, one per framework sample made
  from this episode. verl produces one sample per episode; miles' session server can produce
  several (one per kept leaf path). Each attestation records the framework sample id, the
  terminal response it covers, whether the complete terminal span survived truncation, the
  token count, loss-token count, a hash of token ids and loss mask together, log-probability
  coverage of the loss tokens, and the eligibility disposition. Raw token arrays stay with the
  framework. "Retained by collection" is not "consumed by an optimizer update"; the latter is
  recorded by the framework's metrics, not by the episode.

Reward eligibility is decided per attestation:

| Case | Reward |
|---|---|
| terminal answer present and its complete span retained | the scorer's reward |
| no answer (policy failure: max turns, empty or length stop), retained | a valid zero, if the scorer policy says so |
| reward refers to a terminal answer that collection truncated or dropped | invalid; the sample is excluded |
| `score.valid` is false, or `status` is `infra_failure` or `policy_failure` (the driver stopped without a decision) | invalid; the sample is excluded |

Neither framework interprets an invalid reward on its own; the adapter must keep invalid
samples out of the update (zero loss mask or removal) and count them.

Default decision rules (in `agent.policy`, used by every driver). The policy consumes the
driver's parsed calls and the driver's token accounting; it does not tokenize.

- Tool calls in one turn run in order, up to the per-turn limit; the rest are recorded as
  `skipped`. A terminal call beyond the limit is `skipped` too.
- The first well-formed terminal call within the limit ends the episode; every other call in
  that turn is `skipped`, since its result could never reach the model.
- A malformed call, a terminal one included, is returned to the model as a tool error and counts
  toward the limit.
- A length stop ends the episode without an answer, even if a terminal call was parsed from
  the cut-off text. An empty response ends the episode without an answer.
- Budgets are separate: total context tokens (prompt plus everything appended), cumulative
  generated tokens, a per-turn generation reserve (the next turn needs at least this much room),
  and the per-result tool-output cap in bytes.
- The observations of one turn are admitted together, only if, after truncation to the
  tool-output cap, they leave the generation reserve free in the context budget; otherwise they
  are `not_admitted` and the episode ends with `context_budget`.
- A turn's `<think>` reasoning is separated from its visible text the same way by every driver,
  and the answer is read from the visible text. The Chat Completions harness returns the
  reasoning to the server with the turn, so the episode's context matches verl's token context.

## Tools and isolation

Tools implement `call(arguments, context) -> ToolResult` with per-call timeout and output cap.
They run either in-process (tests, small tools) or in a **node-local tool service** (HTTP), which
loads corpora and indexes once per node and is shared by all episodes and frameworks. miles runs
agent functions in child processes by default; the service keeps heavy state out of them. The
service is found through the node of the process that runs the episode, not the driver's node.
The service is started before the framework and stopped after it; a missing service is an
infrastructure failure, not a tool error.

Sandbox levels for code-executing tools:

| Level | Meaning |
|---|---|
| `strict` | separate user, PID and mount namespaces, empty `/proc`, no network, read-only root, private `/tmp`; fails if unavailable |
| `best_effort` (default) | `strict` when available, otherwise a resource-limited subprocess; network isolation is not guaranteed; the fallback is logged |
| `off` | plain subprocess; tests only |

The level reached, and which isolation features were actually in place, are recorded per
tool call and reported in evidence.

As implemented (`rft_cookbook.tools.service`, `rft_cookbook.tools.sandbox`): a task with heavy
tools provides `local_tools(resources)` and optionally `warm(resources)`; the launcher starts
`python -m rft_cookbook.tools.service` for the length of a stage, and the service writes a
discovery file (`/tmp/rft-tools/<run id>.json`) on its own node once its tools are warm. An
episode's `RemoteToolSet` reads that file on the node it runs on. An unreachable service, or a
tool's `ToolInfraError` crossing it, ends the episode as an infrastructure failure. The service
applies the timeout and output cap, so in-process and remote tools give the same text. Sandbox
`strict` uses bubblewrap: new user, network, PID, IPC and UTS namespaces, a read-only root, an
empty `/proc` (AI Runtime forbids a fresh proc mount), a private `/tmp`, and empty tmpfs over
`/Volumes`, `/dbfs`, `/home`, `/root`, `/local_disk0`, `/mnt` and `/databricks`, except a
directory holding the interpreter (on AI Runtime the interpreter is in `/opt/venv`).

## Scoring and judges

`Scorer.score(episode, ground_truth, metadata) -> RewardResult(valid, score, components)`.
A judge scorer is driven by a versioned **judge policy**:

- rubric version and request format (what part of the episode the judge sees, and its budget);
- allowed models, in order; whether a fallback model's labels may enter headline evaluation;
- concurrency limits, timeouts, retries. A limit that must hold for a whole endpoint, across
  rollout processes and nodes, is enforced by the judge client in the tool service, not by
  per-process locks;
- failure handling: an unresolved verdict marks the episode `infra_failure`, not a zero reward.

The judge identity and verdict are stored in the episode record. Every allowed judge model is
calibrated on the task's fixtures before use.

## Framework adapters

Each adapter implements: data conversion, tool registration, the episode driver, the reward
hook, the retention attestation, checkpoint publishing and the launch command.

**verl** (`frameworks/verl`)

- Episode driver: a subclass of verl's tool agent loop that calls `agent.policy` and emits
  `EpisodeRecord`s. This native loop is the supported default. The Uni-Agent gateway path
  (Chat Completions harness) is added after it matches the native loop on a fixed checkpoint.
- Trainer: verl's V1 trainer (`trainer.use_v1=True`, `trainer.v1.trainer_mode=separate_async`,
  `actor_rollout_ref.hybrid_engine=False`, standalone rollout pool, a non-naive checkpoint
  engine). It provides the post-save callback (`trainer.checkpoint_callback_class`). The
  deprecated `experimental.fully_async_policy` path is not used.
- Reward: the agent loop scores the finished `EpisodeRecord` with the task's scorer and sets
  `AgentLoopOutput.reward_score`, which bypasses verl's reward loop; training and evaluation score
  the same record with the same code. An excluded sample (infrastructure failure, invalid score,
  truncated terminal turn) raises: verl's V1 replay buffer then drops the whole prompt group and
  refills it, so the sample neither enters the update nor biases its group's baseline. verl refills
  without limit, so a broken dependency would spin without training: after
  `train.max_consecutive_exclusions` failed episodes in one agent-loop process without a decided
  one between them (a loop that cannot be built, or a prompt that leaves no room for a turn,
  counts; a valid zero that needed no judge neither counts nor ends the streak, since in a judge
  outage those still score), the loop writes the run's `train/ABORT.json` and the launcher stops
  verl. When the failures are the judge's, rollouts first wait for it (each invalid score is
  retried every minute) for up to `train.judge_outage_wait_s` (`rft_cookbook.agent.guard`).
- Records: each loop writes its `EpisodeRecord`s as closed JSONL chunks to the run's volume.

**miles** (`frameworks/miles`), upstream only. Hook paths are dotted (`module.attr`); miles
splits them at the last dot.

| Need | Upstream extension point |
|---|---|
| episode driver | upstream's `--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate` with `--custom-agent-function-path rft_miles.agent.run`, run inline (`--custom-agent-function-mode inline`) on session server v2: the Chat Completions harness against the session's endpoint, then the task's scorer |
| replay matching | `--session-message-matcher rft_miles.session.matches`: accepts a replayed assistant turn that is the harness's reading of the stored one (reasoning split, only the answered calls), so a harness episode stays one lineage and one sample |
| reward | `--custom-rm-path rft_miles.reward.reward`: the score the agent function returned, which session server v2 merges into every sample of the session; an excluded episode has none, and miles drops its group |
| data | prompt rows (`--prompt-data`) carrying the cookbook row and the run's agent settings in `metadata` |
| checkpoint publishing | `--save`, `--load`, `--save-interval`, `--save-hf`, `--custom-megatron-post-save-hook-path rft_miles.checkpoint.publish` (rank 0, synchronous) |
| model support | the model's launcher (`model.launcher`): upstream's where it fits (GLM-5.3-Flash), the cookbook's own under `rft_miles/launchers/` where a recipe must match a verl recipe (Qwen3.6-35B-A3B) |
| retention attestation | `--session-sample-postprocessor-path rft_miles.attest.postprocess` (session server v2): miles' default post-processing, then one attestation per sample; a sample cut inside its terminal answer loses its reward. The samples are built after the agent function wrote its record, so the attestations are lines of their own under `<episode dir>/attestations`, with their episode id, not the record's `collection` |

Session server v2 rules out `--group-rm`, `--recompute-logprobs-via-prefill` and partial rollout;
no recipe uses them. Upstream's GLM-5.3-Flash launcher turns off optimizer and RNG saving when it
saves itself; the cookbook leaves its saving off and passes its own `--load` and `--save`, so every
bundle holds the full state.

Failure transport. Upstream generation catches agent exceptions and still collects partial
samples, so cookbook exception types are not enough. The agent function returns the score with
the episode's status; a sample without a score (the function raised, or its return was lost) gets
no reward, so its group is dropped, and a raising function counts toward
`train.max_consecutive_exclusions` like any excluded episode, under the same guard as verl's loop.

Adapters are the only code that imports a framework. Core modules must import without any
framework installed.

## Checkpoints

A post-save hook does not mean a checkpoint is complete or durable: verl may still be writing
asynchronously, resume needs more than the model directory, and miles saves rollout state in a
separate directory, before or after the model depending on the training loop. So each adapter
describes a checkpoint as a **bundle**: the required artifacts, where each lives, the native
completion condition of each, and what the bundle can resume (model, optimizer, RNG, data
position, in-flight rollout state, Hugging Face weights).

`CheckpointStore` gives every framework the same operations on bundles:

- **publish:** copies every required artifact to the Unity Catalog volume, verifies sizes and
  hashes against a manifest, and writes the completion marker last, only after the whole bundle
  verifies. The bundle's Hugging Face weights are checked too; what is missing is recorded in the
  manifest and the result (`hf_problems`) without failing it, since the bundle still resumes. It
  returns a result (`published | failed`, with the reason). While a copy runs, its
  source is protected from the framework's own retention. Training waits on the result at the
  next save and at exit (`drain`).
- **latest:** the newest bundle with a completion marker; partial copies are ignored;
- **resume:** returns the framework-specific load arguments for that bundle, after copying it back
  to local disk if needed;
- **export:** locates the Hugging Face weights the framework wrote and verifies them (complete
  files, loadable config, hash). A missing or failed export fails the operation; miles logs
  periodic export errors without failing, so its marker is checked explicitly;
- **retention:** keeps the last N, every bundle the caller names (an evaluated one), and the
  newest bundle whose Hugging Face weights were complete. verl's hook applies `checkpoint.keep`
  from its configuration; miles' hook cannot be configured, so its train stage records the number
  in the store (`set_retention`) and the hook reads it there.

Checkpoints are written to a filesystem shared by all trainer ranks, or each node publishes its
own shards; a rank-0 hook never assumes it can see other nodes' local files.

A recipe chooses where the framework saves (`checkpoint.location`). `local`: on the node's local
disk, published as a copy (the 2B quickstart: 31 GB a save, ~5 minutes). `volume`: straight into
the store, published **in place**: the bundle's root is its published location, nothing is
copied, and the manifest and completion marker are added to the native directory once the
framework reports the save durable. On several nodes only `volume` resolves: verl's trainer pool
can land on another node than the driver that runs the hook, and TransferQueue saves its state
from every node into one directory. The 35B OfficeQA recipe saves ~490 GB each time (70 GB of
Hugging Face weights, 420 GB of optimizer state), so it records sizes, not hashes
(`hash_files: false`), and keeps the newest two (`keep: 2`), as its miles twin does (555 GB a save);
neither framework deletes a save itself.

Milestone 1 uses synchronous native saves and synchronous publication. Background publication
follows once the lifecycle above is tested.

| | verl (V1 trainer) | miles |
|---|---|---|
| save | native (`trainer.save_freq`), synchronous at first | native (`--save`, `--save-interval`) |
| hook | `trainer.checkpoint_callback_class` (driver; `async_save=True` means not yet durable) | `--custom-megatron-post-save-hook-path` (rank 0, after native and HF save) |
| bundle | `global_step_N/{actor, data.pt, transfer_queue}` + `latest_checkpointed_iteration.txt` | Megatron `iter_N` directory, with the HF weights inside (`iter_N/hf`) |
| HF weights | verl checkpoint manager with `hf_model` in `save_contents` | `--save-hf` |
| resume | `trainer.resume_mode=resume_path` + path | `--load` |

## Evidence and qualification

Shared by every agentic task (`rft_cookbook.evidence`):

- **evaluation plan:** question ids × samples, frozen before the run;
- **completeness:** every planned episode is present and either scored or explicitly an
  infrastructure failure; thresholds per recipe;
- **leakage:** training and evaluation ids are disjoint, by id and by content hash;
- **identity:** the evaluated checkpoint matches a published manifest; base model and tokenizer revisions;
- **accounting:** infrastructure failures, judge failures, fallback labels, sandbox levels;
- **comparison:** paired per-question differences with bootstrap intervals resampled by
  question (several samples per question are not independent pairs); McNemar only with one
  sample per question; seeds reported. A replicate trains with its own `train.seed`, samples its
  evaluation with its baseline's seed (`eval.seed`) and is compared with that baseline
  (`eval.baseline_run`) rather than evaluating a new one;
- **qualification record:** code commit, recipe hash, resolved configuration hash, image digest,
  model and tokenizer revisions, data, corpus and index identity, evaluation policy, topology,
  and the checks passed.

A recipe's status is **measured**, **smoke-tested** or **untested**, tied to a qualification
record. Changing the task, agent policy, scorer, adapter, launcher or image invalidates it.
Results produced by earlier implementations are linked as history, not as the recipe's status.
The `compare` stage writes the record (`rft_cookbook.evidence.qualification`, the run's
`qualification.json`): **measured** when every check passed on the recipe as checked in (a training
or evaluation seed, the baseline run and the stages may differ), **smoke-tested** when they passed
on a reduced run. No recipe has a record yet: until then the README calls a recipe **done** when its
GPU runs passed (its header lists them), and **untested** otherwise.

Comparisons across frameworks are recipe comparisons: each reports its paired gain over its own
baseline, with compute and judge cost, unless model, data, budgets, judge policy, evaluation
serving and training budget are all matched.

## Layout

```
ai-runtime-rft-cookbook/
├── README.md                     # support matrix: task × framework × model → status, result
├── pyproject.toml                # package rft-cookbook; light dependencies only
├── site.example.env              # workspace settings for `air run --override` (site.env is gitignored)
├── src/rft_cookbook/
│   ├── agent/                    # spec.py, policy.py, harness.py, episode.py, errors.py
│   ├── tools/                    # base.py, service.py (node-local tool service), sandbox.py
│   ├── scoring/                  # base.py, judge.py
│   ├── tasks/
│   │   ├── officeqa/             # first agentic task
│   │   └── math/                 # quickstart
│   ├── eval/                     # serve a checkpoint, run the harness, score, write evidence
│   ├── evidence/
│   ├── checkpoints/
│   ├── recipe.py                 # recipes and `extends`
│   └── air/                      # job plan, the shared stages, Ray cluster, model staging
├── frameworks/
│   ├── verl/
│   │   ├── rft_verl/             # agent_loop, data, checkpoint callback, render, launch
│   │   ├── launch/               # node entry (node.sh)
│   │   ├── image/                # Dockerfile, locks, patches with removal conditions
│   │   ├── tests/
│   │   └── README.md             # pinned versions, supported modes, known issues
│   ├── miles/
│   │   ├── rft_miles/            # agent, session, reward, checkpoint, render, launch, launchers/
│   │   ├── launch/
│   │   ├── image/                # miles' reference image moved onto the AI Runtime base
│   │   ├── tests/
│   │   └── README.md
│   └── slime/                    # later
├── recipes/
│   ├── agentic/{verl,miles}.yaml # the generic agentic recipe per framework
│   ├── math/verl/qwen3.5-2b_h100x8/
│   ├── math/miles/{glm-5.3-flash-4layer_b300x8, glm-5.3-flash_b300x32}/
│   ├── officeqa/verl/{qwen3.5-35b-a3b_h100x16, qwen3.6-35b-a3b_h100x16}/
│   └── officeqa/miles/{qwen3.6-35b-a3b_h100x16, glm-5.3-flash_b300x64}/
├── docs/
└── tests/
```

A recipe extends the generic agentic recipe (abridged from `recipes/officeqa/verl/qwen3.6-35b-a3b_h100x16`):

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

Recipes are configuration and reference hooks by import path. A recipe may carry an
experimental hook under `hooks/` with a test; a hook used by a second recipe moves into its
task or adapter.

`extends` resolution: the parent path is relative to `recipes/`; mappings merge recursively,
lists and scalars from the child replace the parent's; command-line overrides apply last.
Unknown keys and cycles are errors. The resolved configuration is written out and hashed; the
hash goes into the run's plan and its qualification record.

## Stages

A recipe lists its stages in the order they run (`stages`): `prep` (the task's rows, and its
corpus and index), `stage_model` (the model at its pinned revision on the volume, converted where
the framework needs it), `stage_judge` and `calibrate` (a judged task: the judge's tokenizer, then
the judge grading known cases before it grades anything else), `baseline_eval`, `parity` (the base
model's evaluation against a recorded reference, where a recipe has one), `train`, `export` (the
newest published checkpoint's Hugging Face weights, checked), `eval` and `compare`. Each stage reads
what the earlier ones left under the run's directory on the volume; nothing inserts or skips a
stage. Training runs on the recipe's nodes, model staging and evaluation on one node of its
accelerator, the other stages on one A10.

Each stage of a recipe is one AI Runtime job, checked in next to the recipe as
`jobs/<stage>.yaml`: the recipe it runs (`env_variables.RFT_RECIPE`), its compute, timeout and
launcher command, and placeholders for the run id, volume root, image and secrets that the user
fills in with `air run --override`. Nothing is generated: inside the job the launcher resolves
the recipe from the code snapshot (with `RFT_SET` overrides) into the run's files (Hydra
arguments, agent-loop registry, plan of paths). A test keeps every checked-in job consistent with
its recipe (stages, compute, timeouts, command).

## Packaging

`rft-cookbook` and each adapter (`rft-verl`, `rft-miles`) are small Python packages. At job
start they are installed from the uploaded code snapshot with `pip install --no-deps` on every
node, into the interpreter the framework uses, before Ray or any worker starts. Every process,
including Ray workers and miles agent subprocesses, then imports the same code without path
manipulation. `--no-deps` does not install missing dependencies, so the packages depend only on
what the framework images already contain, and they build with a pinned backend that needs no
network. Package data (prompts, tool schemas, recipe defaults) ships inside the packages.
Frameworks are never declared as dependencies; they come only from their images. A verl user
never installs miles or slime. Tests import the installed packages from outside the checkout
and inside a Ray worker.

## Testing

| Tier | What | Where | When |
|---|---|---|---|
| 1. Unit | agent policy rules (multiple calls, terminal at limits, malformed calls, empty and length stops, budget admission); tools on fixtures; harness against a fake chat server; judge policy against fake endpoints; sandbox level detection; evidence checks; episode schema | GitHub Actions, CPU | every PR |
| 2. Adapter contract | each adapter turns recorded fixtures into valid `EpisodeRecord`s; one recorded episode scores identically through every adapter; attestations on truncated, multi-leaf and retried sessions; invalid rewards never reach the update; lost agent metadata becomes `infra_failure` | GitHub Actions, CPU, fake framework objects | every PR |
| 3. Framework API | adapters against the real pinned framework: hook signatures, base classes, configuration composition | maintained runner with the framework image | changes to an adapter, core, task or pin |
| 4. Jobs and public safety | every recipe resolves; its checked-in jobs match it; secret and workspace-detail scan of sources, jobs, JSON, SVG and evidence | GitHub Actions, CPU | every PR |
| 5. GPU smoke | per framework: small model, two updates, episode records valid, retention attested, checkpoint published, resumed and exported, export evaluated | AI Runtime, maintainer-triggered from a reviewed commit | pin, image, adapter, core or task changes; before releases |
| 6. Full recipe | the headline result with a qualification record | AI Runtime | when a result must be established |

GPU tiers never run on pull requests from forks. Their sanitized receipts are published with the
qualification record.

## Public repository rules

- `site.env` holds the workspace profile, volume root, images and secret references (never
  values) used to fill in a job's placeholders at submission. It is gitignored and excluded
  from code snapshots.
- CI scans for credentials, e-mail addresses, workspace hosts, user paths
  (`/Volumes/<catalog>/<user>`, `/Workspace/Users/...`) and run URLs.
- Published evidence contains aggregates and per-question scores with ids; it records what was
  removed during sanitization and the hash of the original artifact.
- Datasets and corpora are downloaded at preparation time; traces published as examples follow
  each dataset's redistribution terms.
- New code is under the repository license. Files derived from Apache-2.0 sources keep the Apache
  license with attribution and a modification notice; the README lists them.

## Milestones

1. **Agentic core on verl, math quickstart.** Agent specification, policy, episode format,
   in-process tools, scoring, evidence basics, `CheckpointStore` with the verl V1 callback
   (synchronous). `recipes/math/verl/qwen3.5-2b_h100x8` (4 trainer + 4 rollout GPUs): prepare →
   two updates → publish → restart → resume and run a further update → publish → export → serve
   that export → evaluate it through the harness with complete episode accounting. The aim is
   plumbing, provenance, restart and export identity, not a learning result. A recorded OfficeQA
   episode passes through the scorer.
2. **OfficeQA on verl** through the generic agentic recipe: tool service, sandbox levels, judge
   policy, completeness checks. The base model's evaluation must agree with the earlier baseline
   within a stated tolerance (paired by question) before any training run; the decision rules
   differ slightly from the earlier driver, so an identical number is not expected.
   The recipe states the tolerance: for supported pass@1 and answer accuracy, on the 43 held-out
   questions and on the 16 training probes, every reference question is evaluated and the paired
   question-bootstrap 95% interval of the difference contains zero (`parity` stage,
   `reference_baseline.json`). Known differences from the reference driver (verl's agent loop
   against vLLM, continuous tokens): the Chat Completions harness renders turns through the
   server's chat template; the tool-result cap is 8,000 UTF-8 bytes of tool text, not 8,000
   characters including the `[observation obs_N]` line; a terminal call parsed from a turn that hit
   the per-turn token limit is not accepted (the reference accepted it if it fit the response
   budget); the next turn needs 64 free tokens. Per-episode sampling seeds follow the reference
   (`seed + question index × samples + sample`).
   One difference is the reference's, not ours: verl 0.10's `ToolAgentLoop` validates tool schemas
   into `OpenAIFunctionToolSchema`, which keeps only `type`, `description` and `enum` of each
   parameter, so the reference model never saw the fields of `submit_report`'s `path` items and
   often swapped them, invalidating its own citations. The cookbook shows the full schemas in
   evaluation and training. `eval.schema_view: verl-0.10-properties` reproduces the reference's
   view for a diagnostic evaluation. With the full schemas the base model scores 15.1% held-out
   against the reference's 8.7%, and `parity` disagrees; parity is therefore checked in a run of its
   own with the reference's view (`RFT_SET="eval.schema_view=verl-0.10-properties"`, stages
   `baseline_eval` and `parity`), where every metric agreed (held-out supported pass@1 13.4%
   against 8.7%, +4.7 points [−0.6, +10.5]). The training run skips `parity`.
   Training on two nodes: every node runs the task's tool service and joins one Ray cluster
   (`rft_cookbook.air.cluster`: the head waits for every GPU, records its outcome in the run's
   rendezvous directory, and workers leave with it, or with a failure when the head stops
   heartbeating); verl runs on the head.
   Result (2026-10-22, `officeqa-35b-m2-01`, one training seed): 100 updates in about 7.5 hours,
   every save published in place; held-out supported pass@1 15.1% → 22.1%, +7.0 points, paired
   question-bootstrap 95% interval [+1.2, +13.4] (43 questions × 4 samples, every episode valid,
   leakage clean); the 16 training probes 39.1% → 56.3%. The gain is mostly evidence the judge
   supports: answer accuracy alone moves +3.5 points [−2.3, +9.9]. The training score stays flat
   (16 trajectories per update). The reference run reached 20.9% from its 8.7% baseline: the
   trained models are close, and the cookbook's baseline is higher because its model sees the
   full tool schemas. The replicate with training seed 7 (`officeqa-35b-m2-03-seed7`, evaluated
   against the same baseline with the same evaluation seed) reproduces the held-out gain: 15.1% →
   20.9%, +5.8 points [0.0, +12.2]. The probe gain does not replicate (39.1% → 39.1%, +0.0
   [−10.9, +12.5]), and the replicate ends more episodes on the context budget (31 against 16), so
   two seeds support a held-out improvement of about six points, not its size. The replicate was
   also stopped after its 25th update and resumed from the volume on two nodes.
3. **OfficeQA on miles (upstream)** through the same recipe: the 4-layer GLM-5.3-Flash slice
   first, then the full topology. One model on both frameworks gives a matched comparison:
   Qwen3.6-35B-A3B has a verl recipe and a miles recipe with the same data, budgets, judge,
   evaluation and training settings, wherever miles allows them (the miles recipe lists the
   differences: rollouts one update behind where verl's may lag by up to four, SGLang serving the
   training rollouts, data order; both train on 8 H100s and serve on 8 others). The stages that do
   not train run the same code for both (`rft_cookbook.air.stages`): every framework's checkpoint
   is exported to Hugging Face weights and served by vLLM through the same harness, so the miles
   run is compared with the verl run's baseline evaluation. An excluded episode drops its group on
   both (verl's trainer refills it; miles drops a group with a sample without reward and samples
   another prompt). The full GLM-5.3-Flash trains every parameter in upstream's colocated layout:
   the math task on four 8xB300 nodes, then OfficeQA's 98K-token episodes on eight, as a
   measurement. Upstream cannot split
   GLM-5.3's sequences over GPUs (no context parallelism in its DSA indexer), which a 150K-token run
   of this model has needed.
4. **Agentic search** as the second full task; generalize what OfficeQA hard-coded. Gateway spike
   on verl, compared with the native loop on a fixed checkpoint.
5. **slime**, then other frameworks on demand.
