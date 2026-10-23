# miles adapter

`rft_miles` runs cookbook tasks on upstream miles through miles' own extension points:

| Need | miles extension point | Module |
|---|---|---|
| episode driver | `--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate` with `--custom-agent-function-path` (inline) on session server v2: the cookbook's Chat Completions harness drives the episode against the session's OpenAI endpoint; the session server records tokens and log-probabilities | `agent.run` |
| replay matching | `--session-message-matcher`: the harness replays a turn as its policy read it (reasoning split from the text, only the calls it answered); the matcher accepts exactly that reading of the stored message, so an episode stays one lineage and trains as one sequence, as on verl | `session.matches` |
| reward | `--custom-rm-path`: the score the agent function returned, merged into every sample of the session; an excluded episode has none, and miles drops its group and samples another prompt | `reward.reward` |
| checkpoint publishing | `--custom-megatron-post-save-hook-path` (rank 0, after the save); `--save` is the store, so the bundle is published in place | `checkpoint.publish` |
| data | prompt rows (`--prompt-data`) carrying the cookbook row and the run's agent settings in `metadata` | `launch` |
| launch | the model's launcher (`model.launcher`): upstream's (e.g. `scripts/run_glm5_3_flash.py`) or the cookbook's own (`rft_miles/launchers/`), with the cookbook's flags in its `--extra-args` | `render`, `launch`, `launch/node.sh` |
| tools | the task's node-local tool service on every node for the length of the train stage, as on verl | `launch` |
| stop | after `train.max_consecutive_exclusions` excluded episodes in a row the agent function asks the launcher to stop the run | `agent.run`, `launch` |

## Pinned stack

Upstream miles at the commit in `rft_miles/miles.lock.json`, fetched at job start and checked against the
tarball's sha256, then installed editable (its launchers find `train.py` and `tools/` beside the
package). The pin is miles' v0.1.3 release commit (#4275), unmodified. The Megatron conversion of a
model is kept per miles commit.

The image (`image/`) provides miles' dependencies: it is miles' own reference image for that
release, `radixark/miles:release-v0.1.3-ci` (CUDA 13.0, PyTorch 2.14, SGLang's `sglang-miles`
branch, Megatron-LM's `miles-main`), moved onto the AI Runtime base. The reference is 41 GB
unpacked, twice AI Runtime's 20 GB limit, so the image takes its Python environment as built and
leaves out what no cookbook run uses: FlashInfer's prebuilt kernels for every GPU architecture
(FlashInfer compiles what a run uses, with the image's nvcc), FlashAttention 3, packages off the
cookbook's path, static libraries, and miles itself; it patches and swaps nothing (`image/Dockerfile`
lists each). The build checks that the reference was built from the pinned miles commit, so move
the pin and the image together:

```bash
bash frameworks/miles/image/build.sh <registry>/<repository>:<tag>
```

Then push it, set its digest in `site.env` (`RFT_MILES_IMAGE`) and register it with AI Runtime. The
AI Runtime base lacks the system libraries of a few optional pieces of the reference: SGLang's
MSCCL++ all-reduce (GDRCopy), NVSHMEM's PMIx, SHMEM and UCX bootstraps, and video decoding. The image
adds what the cookbook's tasks need, as the verl image has it: `bm25s` (OfficeQA's retrieval),
`databricks-sdk` (the hosted judge's credentials) and bubblewrap (the strict compute sandbox), the
Python packages without their dependencies so that the reference's own stay as they are.

## Model launchers

A recipe names the launcher that trains its model. Upstream's are used as they are where they fit
(GLM-5.3-Flash). Where a recipe must match a verl recipe of the same model and upstream's launcher
cannot (it switches on features that a later flag cannot switch off), the cookbook carries its own,
derived from upstream's and named after the model:

| launcher | from upstream's | differs in |
|---|---|---|
| `rft_miles/launchers/qwen3_6_35b_a3b.py` | `scripts/run_qwen3_6_35b_a3b_mtp.py` | no MTP training or EAGLE decoding; verl's algorithm defaults (token-mean loss, dual-clip 3.0, weight decay 0.01, Adam beta2 0.999, MoE auxiliary loss 1e-3); verl's parallelism (TP2, CP4, EP8); OpenAI sampling defaults; any 8-GPU node; no download or conversion; the GLM launcher's options |

## Evaluation

Evaluation does not depend on the framework that trained the checkpoint (`rft_cookbook.air.stages`):
`train.save_hf` writes Hugging Face weights into every saved bundle (`--save-hf`, with miles'
weight-sync converters), `export` checks them, and `baseline_eval` and `eval` serve them with vLLM
from the verl image, through the same harness, sampling and scorer as a verl run. Those jobs install no
miles. A miles run of a model that also has a verl run can therefore be compared with the verl run's
baseline evaluation (`eval.baseline_run`).

## Supported mode

Session server v2, on one node or several, in the layout of the model's launcher: upstream's
GLM-5.3-Flash launcher colocates training and SGLang on the same GPUs; the cookbook's Qwen3.6
launcher trains on half the nodes and serves on the other half (`train_async.py`, as verl's
separate_async), so neither shares a GPU with the other. On several nodes every node copies the
model to local disk and writes the agent inputs, then node 0 runs miles on one Ray cluster of all of them
(`rft_cookbook.air.cluster`, with `MILES_SCRIPT_EXTERNAL_RAY=1`) while the others serve as workers.
The upstream launcher validates the node and GPU counts it supports for a model.

## Known limits

- **Staleness differs from verl's.** A colocated launcher samples every rollout with the current
  weights; `train_async.py` (Qwen3.6) samples each with the weights one update behind; verl's
  separate_async rollout may lag by up to `train.max_off_policy_threshold` versions.
- **A branch is a sample.** Session server v2 makes one sample per branch of a session, each with
  the episode's reward. With the cookbook's matcher a harness episode is one branch, as verl makes
  one sample per episode; a request that does not continue the stored history (a retry after a
  server error) still opens another. Without it, a model whose reasoning the server leaves in the
  text (Qwen3.6 here) or a turn with more calls than `max_tool_calls_per_turn` would branch at
  every turn: a first Qwen3.6 smoke made 499 samples of 16 episodes, each with its whole prefix.
- **The source is fetched at job start.** GitHub generates the tarball; if its bytes ever change,
  the sha256 check fails the job rather than run other code.
- **Every save stays on the volume.** The adapter publishes each bundle in place but does not prune
  them yet (verl's `checkpoint.keep` has no miles counterpart): a Qwen3.6 run keeps all four saves.
- **No retention attestation yet.** miles' episode records have no `collection` entries; the session
  sample postprocessor that would write them is not implemented.
