# verl adapter

`rft_verl` runs cookbook tasks on [verl](https://github.com/verl-project/verl), using only verl's
own extension points.

## How it plugs into verl

| What | verl extension point | Module |
|---|---|---|
| Running episodes | A `ToolAgentLoop` subclass, registered through `rollout.agent.agent_loop_config_path` | `agent_loop.CookbookAgentLoop` |
| Reward | The loop scores each finished episode and sets `AgentLoopOutput.reward_score`, so verl skips its own reward loop | `agent_loop` |
| Excluding a sample | The loop raises; verl's V1 replay buffer drops the prompt group and refills it | `agent_loop` |
| Publishing checkpoints | `trainer.checkpoint_callback_class` (V1 trainer) | `checkpoint.PublishCheckpoint` |
| Data | Parquet for `RLHFDataset`, whose `agent_name` routes each row to the cookbook loop | `data` |
| Launch | `python -m verl.trainer.main_ppo` with the recipe's Hydra overrides, resolved inside the job | `render`, `launch`, `launch/node.sh` |

## Supported setup

- verl's V1 trainer in `separate_async` mode: training runs on `compute.trainer_gpus`, and a
  separate vLLM pool generates rollouts on `compute.rollout_gpus`. Weights go to the pool over
  NCCL after each update.
- The Megatron engine, through Megatron-Bridge, without dynamic batch sizes. Packed sequences are
  used only where the recipe says the model supports them (`strategy.remove_padding`: on in the 35B
  MoE recipes, off in the 2B quickstart).
- One node, or several. With several nodes, each GPU pool fills whole nodes and checkpoints are
  saved straight to the volume.

Resolving a recipe fails for anything else. The older `experimental.fully_async_policy` path is not
used.

## Pinned versions

verl 0.10.0.dev at commit `8718ca30a3f002f93b7c4fd99b9b2506718681bc`, PyTorch 2.13, vLLM 0.30,
Transformers 5.16.1, Megatron-Bridge 0.6.2, Megatron-Core 0.19.2 and Transformer Engine 2.16.1.
The full lists are in `image/artifacts.lock` and `image/requirements.lock`.

The image applies one patch, `image/patch_verl_weight_chunks.py`. The pinned Megatron-Bridge
exports some Qwen3.5 tensors with strides, and verl's weight-sync serializer needs them
contiguous. The patch refuses to apply to any verl source but the pinned one. Remove it when the
pin moves past the upstream fix and `check_verl_weight_chunks.py` passes without it.

## Building the image

```bash
bash frameworks/verl/image/build.sh <registry>/<repository>:<tag>
```

The image is for linux/amd64 and is about 17 GB. Push it, put its digest in `site.env`
(`RFT_VERL_IMAGE`), and register it with AI Runtime. The miles recipes use this image too, for
their evaluation stages.

### Behind a proxy or a private package index

The build downloads packages from PyPI and a few pinned wheels and source trees from GitHub. If
your network needs it:

- `PIP_INDEX_URL=<index> bash frameworks/verl/image/build.sh ...` uses your package index. The
  URL is passed as a BuildKit secret, so it isn't recorded in the image history.
- `BUILD_ARGS="--build-arg HTTPS_PROXY=http://proxy:3128"` passes extra flags to `docker build`.
- Behind a TLS-intercepting proxy, put the proxy's CA bundle (`*.crt` or `*.pem`) in
  `frameworks/verl/image/certs/`. The image then trusts it. The folder is gitignored.
- If the build can't reach GitHub, download the pinned artifacts on the host into
  `frameworks/verl/image/vendor/` (gitignored), and the build uses them instead:
  - `vendor/wheels/<file>`: the wheels listed in `image/artifacts.lock`, under the file name of
    their URL. Each is checked against the sha256 in the lock.
  - `vendor/src/<name>/`: the source trees of `megatron-core` and `verl` at their pinned commits,
    each with its commit hash in a `.voa-commit` file. A tree at any other commit stops the build.

## Known limits

- **Checkpoints are saved synchronously.** With asynchronous saves, verl calls the post-save hook
  before the checkpoint is on disk, so `PublishCheckpoint` refuses `checkpoint.async_save`.
- **One failure drops a whole group.** If one episode of a GRPO group fails (an infrastructure
  error or an invalid score), the whole group leaves the update and verl refills it. The
  episode records and verl's `failure` counts show how often this happens.
- **A broken setup retries until stopped.** Because verl refills failed groups, an error
  that fails every episode (for example, data prepared for another task version) would repeat
  until the job times out. `prep` checks that the data carries the task's prompt, and the loop
  stops the run after `train.max_consecutive_exclusions` failures in a row.
- **verl's own `ToolAgentLoop` shows the model simplified tool schemas.** It keeps only the
  `type`, `description` and `enum` of each parameter, so nested fields disappear. The cookbook
  loop shows the full schemas. `eval.schema_view: verl-0.10-properties` reproduces verl's view for
  diagnostics.
- **The drivers parse differently.** verl's tool-call parsers drop calls they can't parse, while
  the Chat Completions harness used for evaluation sees whatever the serving parser returns. The
  decisions after parsing are the same, as
  [`tests/test_drivers_agree.py`](tests/test_drivers_agree.py) checks.
