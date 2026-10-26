# Training with verl

The `rft_verl` adapter runs cookbook tasks on [verl](https://github.com/verl-project/verl).
It uses verl's native tool-agent loop and checkpoint callback. See the
[tested recipes](../../README.md#tested-recipes) for supported models and hardware.

## Supported configurations

Training uses verl's V1 trainer in `separate_async` mode, with a Megatron trainer and a separate
vLLM rollout pool. Weights are sent to the rollout pool over NCCL after each update.

The adapter supports one node or multiple nodes. On multiple nodes, each pool occupies whole
nodes and checkpoints are saved directly to the volume. Packed sequences are enabled for the
35B MoE recipes and disabled for the 2B quickstart. Asynchronous checkpoint saves are not supported.

## Versions

The image pins verl at `8718ca30a3f002f93b7c4fd99b9b2506718681bc`, with PyTorch 2.13,
vLLM 0.30, Transformers 5.16.1, Megatron-Bridge 0.6.2, Megatron-Core 0.19.2, and Transformer
Engine 2.16.1. See [`artifacts.lock`](image/artifacts.lock) and
[`requirements.lock`](image/requirements.lock) for the full dependency pins.

A small [weight-serialization patch](image/patch_verl_weight_chunks.py) makes strided tensors
contiguous before transfer to vLLM. It checks the pinned source before applying. The associated
[regression check](image/check_verl_weight_chunks.py) determines when the patch can be removed
with a future upstream update.

## Building the image

```bash
bash frameworks/verl/image/build.sh rft-verl:<tag>
databricks air images push -p <profile> --source rft-verl:<tag> --catalog <catalog> --schema <schema>
```

The image targets linux/amd64 and is about 17 GB. Set `RFT_VERL_IMAGE` in `site.env` to the
uploaded Unity Catalog image name, and `RFT_VERL_IMAGE_DIGEST` to its digest. miles recipes also
use this image for evaluation.

### Private indexes and proxies

- Set `PIP_INDEX_URL` when invoking the build to use a private package index. It is passed as a
  BuildKit secret, not stored in image history.
- Pass proxy options through `BUILD_ARGS`, for example
  `BUILD_ARGS="--build-arg HTTPS_PROXY=http://proxy:3128"`.
- For a TLS-intercepting proxy, put its CA bundle (`*.crt` or `*.pem`) in `image/certs/`.
- For offline artifacts, place pinned wheels under `image/vendor/wheels/` and pinned source
  trees under `image/vendor/src/<name>/`. Each source tree needs a `.voa-commit` file containing
  its revision. The build verifies wheel hashes and source revisions against its locks.

The certificate and vendor directories are gitignored. Do not commit private certificates or
package-index credentials.

## How the adapter works

| Responsibility | Extension | Implementation |
|---|---|---|
| Run and score an episode | `ToolAgentLoop` subclass, registered in the agent-loop config | [`agent_loop.py`](rft_verl/agent_loop.py) |
| Exclude an invalid sample | V1 replay-buffer group exclusion | [`agent_loop.py`](rft_verl/agent_loop.py) |
| Publish a checkpoint | `trainer.checkpoint_callback_class` | [`checkpoint.py`](rft_verl/checkpoint.py) |
| Prepare framework inputs | Canonical task rows to `RLHFDataset` parquet | [`data.py`](rft_verl/data.py) |
| Resolve and launch a recipe | Hydra overrides for `verl.trainer.main_ppo` | [`render.py`](rft_verl/render.py), [`launch.py`](rft_verl/launch.py) |

The framework owns token IDs, loss masks, and log-probabilities. The adapter records the episode
that received the reward and checks that collection retained its final answer.

## Known limitations

- **An invalid sample drops its group.** verl refills the whole GRPO group. Repeated exclusions
  stop the run instead of allowing an unhealthy dependency to retry indefinitely.
- **Parsing depends on the driver.** verl's parser can drop malformed calls that a serving
  endpoint exposes differently. The shared policy makes the same decisions after parsing;
  [driver-agreement tests](tests/test_drivers_agree.py) cover those decisions.
- **Schema views matter.** The cookbook loop uses full tool schemas. Set
  `eval.schema_view=verl-0.10-properties` only to reproduce the simplified schemas of verl's
  default loop for a diagnostic comparison.

## Testing

The adapter tests need the pinned verl source and a real Qwen3.5 tokenizer, but no GPU.
The `verl` job in [the CI workflow](../../.github/workflows/ci.yml) lists the CPU dependencies
and tokenizer download. With that environment installed:

```bash
pip install --no-deps -e . -e frameworks/verl
RFT_TEST_TOKENIZER=<tokenizer-directory> pytest frameworks/verl/tests
```

These tests exercise the actual `ToolAgentLoop` and Hydra configuration. A GPU smoke test is
still required to check CUDA kernels, weight transfer, and checkpoint behavior in the image.
