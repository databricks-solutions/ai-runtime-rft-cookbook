# verl adapter

`rft_verl` runs cookbook tasks on verl through verl's own extension points:

| Need | verl extension point | Module |
|---|---|---|
| episode driver | agent-loop registry (`rollout.agent.agent_loop_config_path`), a `ToolAgentLoop` subclass | `agent_loop.CookbookAgentLoop` |
| reward | the loop scores each finished `EpisodeRecord` and sets `AgentLoopOutput.reward_score` (verl's reward loop is then skipped) | `agent_loop` |
| exclusion | an excluded episode raises; the V1 replay buffer drops the whole prompt group and refills it | `agent_loop` |
| checkpoint publishing | `trainer.checkpoint_callback_class` (V1 trainer) | `checkpoint.PublishCheckpoint` |
| data | parquet for `RLHFDataset`, with `agent_name` routing rows to the cookbook loop | `data` |
| launch | `python -m verl.trainer.main_ppo` with the recipe's Hydra overrides, resolved inside the job | `render`, `launch`, `launch/node.sh` |

## Pinned stack

verl 0.10.0.dev at commit 8718ca30a3f002f93b7c4fd99b9b2506718681bc, torch 2.13, vLLM 0.30,
Transformers 5.16.1, Megatron-Bridge 0.6.2, megatron-core core_v0.19.2, Transformer Engine 2.16.1
(`image/artifacts.lock`, `image/requirements.lock`). The image applies one patch,
`image/patch_verl_weight_chunks.py`: the pinned Bridge exports strided Qwen3.5 tensors that verl's
weight-sync serializer needs contiguous. The patch refuses any verl source but the pinned one;
remove it when the pin moves past the upstream fix and `check_verl_weight_chunks.py` passes
without it.

Build: `bash frameworks/verl/image/build.sh <registry>/<repository>:<tag>` (linux/amd64, about
17 GB), then push it, pin the digest in `site.env` (`RFT_VERL_IMAGE`), and register it with AI Runtime.

## Supported mode

V1 trainer, `separate_async`, `hybrid_engine=False`: training on `compute.trainer_gpus`, a
standalone vLLM pool on `compute.rollout_gpus`, weights pushed over NCCL; the Megatron engine
through Megatron-Bridge, with packed (THD) sequences only where a recipe says the model supports
them (`strategy.remove_padding`: the 35B MoE recipes; the 2B quickstart runs BSHD) and no dynamic
batch size; one node, or several with each pool on whole nodes and checkpoints saved to the volume.
Resolving a recipe refuses anything else. The deprecated `experimental.fully_async_policy` path is
not used.

## Known limits

- **Synchronous checkpoint saves only.** `PublishCheckpoint` refuses `checkpoint.async_save`:
  with asynchronous saves verl fires the hook before the checkpoint is durable.
- **A misconfigured loop keeps failing groups.** verl refills a failed prompt group, so an error
  that fails every episode (for example data converted for another task version) repeats until
  the job times out. `prep` checks that the converted data carries the task's prompt; the
  agent loop's stop guard (`train.max_consecutive_exclusions`) and the job timeout bound the rest.
- **Excluding a sample drops its group.** One infrastructure failure or invalid score removes the
  whole GRPO group from the update (and verl refills it). The episode records and verl's
  `failure` counts show how often.
- **Driver differences.** verl's parsers drop calls they cannot parse instead of reporting them;
  the Chat Completions harness sees whatever the serving parser returns. The decisions after
  parsing are the same ([`tests/test_drivers_agree.py`](tests/test_drivers_agree.py)).
- **Every recipe is untested** until it has a qualification record. The math recipe's GPU
  acceptance run (two updates, publish, restart, resume, a further update, export, evaluate the
  export) passed on 2026-10-21; the OfficeQA runs are listed in the root README.
