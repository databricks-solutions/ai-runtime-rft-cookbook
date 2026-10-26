# miles adapter

`rft_miles` runs cookbook tasks on upstream [miles](https://github.com/radixark/miles), unmodified,
using only miles' own extension points.

## How it plugs into miles

Each episode runs through miles' session server (v2). The cookbook's Chat Completions harness
plays the episode against the session's OpenAI-compatible endpoint, and the session server
records the tokens and log-probabilities for training. The cookbook's hooks, passed as
command-line flags, provide the rest:

| What | miles flag | Module |
|---|---|---|
| Running episodes | `--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate` with `--custom-agent-function-path` (inline) | `agent.run` |
| Keeping an episode in one sequence | `--session-message-matcher`: accepts the harness's replay of a stored turn (reasoning split from the text, only the calls it answered), so the episode trains as one sequence, as on verl | `session.matches` |
| Reward | `--custom-rm-path`: the score the agent function returned. An excluded episode has none, so miles drops its group and samples another prompt | `reward.reward` |
| Attestations | `--session-sample-postprocessor-path`: miles' default post-processing, then an attestation per sample. A sample cut inside its final answer loses its reward | `attest.postprocess` |
| Publishing checkpoints | `--custom-megatron-post-save-hook-path` (rank 0, after the save). Checkpoints are saved straight to the volume and published in place, keeping the newest `checkpoint.keep` | `checkpoint.publish` |
| Data | Prompt rows (`--prompt-data`) with the cookbook row and the run's agent settings in `metadata` | `launch` |
| Launch | The model's launcher (`model.launcher`), with the cookbook's flags in its `--extra-args` | `render`, `launch`, `launch/node.sh` |

Like verl, the agent function stops the run after `train.max_consecutive_exclusions` excluded
episodes in a row, and holds rollouts for up to `train.judge_outage_wait_s` while a judge is down.
The task's tool service runs on every node during training.

## Pinned version

miles v0.1.3 at its release commit, `8e99947`, unmodified (the release isn't tagged yet). Its
source is fetched at job start, checked against the sha256 in `rft_miles/miles.lock.json`, and installed in editable mode, since its
launchers expect `train.py` and `tools/` next to the package.

## The image

The image is miles' own reference image for that release, `radixark/miles:release-v0.1.3-ci`
(CUDA 13.0, PyTorch 2.14, SGLang, Megatron-LM), moved onto the AI Runtime base image. The
reference image is 41 GB unpacked, twice AI Runtime's 20 GB limit. So the build takes the
reference's Python environment as it is and leaves out what no cookbook run uses: prebuilt
FlashInfer kernels for every GPU architecture (FlashInfer compiles what a run needs), FlashAttention
3, unused packages, static libraries, and miles itself. It patches nothing. `image/Dockerfile`
lists every removal.

The image adds what the cookbook's tasks need: `bm25s` (OfficeQA's retrieval), `databricks-sdk`
(the hosted judge's credentials) and bubblewrap (the strict sandbox). A few optional parts of the
reference don't work on the AI Runtime base, which lacks their system libraries: SGLang's MSCCL++
all-reduce, some NVSHMEM bootstraps, and video decoding.

```bash
bash frameworks/miles/image/build.sh rft-miles:<tag>
databricks air images push -p <profile> --source rft-miles:<tag> --catalog <catalog> --schema <schema>
```

Put the image's Unity Catalog name, `<catalog>.<schema>.rft-miles:<tag>`, in `site.env`
(`RFT_MILES_IMAGE`), and the digest the push printed in `RFT_MILES_IMAGE_DIGEST`. The build checks that the reference image was built from the pinned miles commit, so move
the pin and the image together.

## Model launchers

A recipe names the launcher that trains its model. Upstream's launchers are used as they are where
they fit (GLM-5.3-Flash). Where a recipe must match a verl recipe of the same model, and upstream's
launcher turns on features that can't be turned off later, the cookbook carries its own launcher,
derived from upstream's:

| Launcher | Derived from | Differences |
|---|---|---|
| `rft_miles/launchers/qwen3_6_35b_a3b.py` | `scripts/run_qwen3_6_35b_a3b_mtp.py` | Matches the verl recipe: no MTP training or EAGLE decoding; verl's algorithm defaults (token-mean loss, dual-clip 3.0, weight decay 0.01, Adam beta2 0.999, **no MoE load-balancing loss**); verl's parallelism (TP2, CP4, EP8); training and serving on separate nodes. Also: no download or conversion, and it runs on any 8-GPU node. The launcher's docstring lists every difference |

## Evaluation

Evaluation doesn't use miles. Every saved checkpoint includes Hugging Face weights (`--save-hf`),
the `export` stage checks them, and `baseline_eval` and `eval` serve them with vLLM from the verl
image, through the same harness, sampling and scorer as a verl run. That is why a miles run can be
compared with a verl run's baseline (`eval.baseline_run`).

## Supported setup

Session server v2, on one node or several, in the layout of the model's launcher. Upstream's
GLM-5.3-Flash launcher trains and serves on the same GPUs. The cookbook's Qwen3.6 launcher trains
on half the nodes and serves on the other half (`train_async.py`, like verl's `separate_async`).
On several nodes, every node copies the model to local disk, then node 0 runs miles on one Ray
cluster of all the nodes. The upstream launcher checks which node and GPU counts it supports for a
model.

## Known limits

- **Rollouts are one update behind.** `train_async.py` generates each rollout while the previous
  one trains, so it uses weights one version old. verl's `separate_async` may lag by up to
  `train.max_off_policy_threshold` versions. Colocated launchers always use the current weights.
- **Each branch of a session is a sample.** Session server v2 makes one training sample per branch
  of a conversation. The cookbook's matcher keeps a harness episode in one branch, but a request
  that doesn't continue the stored history (a retry after a server error) still opens another.
  Without the matcher, a model whose reasoning stays in the text (such as Qwen3.6) would branch at
  every turn.
- **The source is fetched at job start.** GitHub generates the tarball. If its bytes ever change,
  the sha256 check fails the job rather than running other code.
- **Attestations are stored beside the episode records.** miles builds a session's samples after
  the agent function has written its record. So each attestation is its own line under
  `<episode dir>/attestations`, with its episode id, instead of the record's `collection` field.
