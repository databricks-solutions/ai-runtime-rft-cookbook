# Training with miles

The `rft_miles` adapter runs cookbook tasks on [miles](https://github.com/radixark/miles).
It uses miles' extension hooks rather than a framework fork. See the
[tested recipes](../../README.md#tested-recipes) to choose a model and GPU layout.

## Supported configurations

The adapter uses session server v2. The cookbook's agent sends Chat Completions requests through
the session server, which captures tokens and log-probabilities for training.

- **Qwen3.6-35B-A3B:** separate training and rollout nodes, using `train_async.py`.
- **GLM-5.3-Flash:** upstream model launchers, with training and serving on shared GPUs.
- **Multiple nodes:** one Ray cluster; every node has a local model copy and, where needed,
  a local tool service.

The source revision and download hash are pinned in
[`miles.lock.json`](rft_miles/miles.lock.json). Job startup verifies the source archive and
installs it in editable mode because the upstream launchers need scripts beside the package.

## Building the image

```bash
bash frameworks/miles/image/build.sh rft-miles:<tag>
databricks air images push -p <profile> --source rft-miles:<tag> --catalog <catalog> --schema <schema>
```

In `site.env`, set `RFT_MILES_IMAGE` to the uploaded Unity Catalog image name and
`RFT_MILES_IMAGE_DIGEST` to its digest. The build checks that the reference image matches the
pinned miles revision; update the image and source pin together.

The image packages miles' reference Python environment on the AI Runtime base. To fit the
20 GB image limit, it omits unused packages and prebuilt kernels for unneeded architectures.
The Qwen launcher uses Triton attention and MoE kernels. The
[Dockerfile](image/Dockerfile) lists the image contents and exclusions.

OfficeQA additionally uses `bm25s`, `databricks-sdk`, and bubblewrap. Optional features such as
MSCCL++ all-reduce, some NVSHMEM bootstraps, and video decoding are not supported by this image.

You also need the [verl image](../verl/README.md#building-the-image) for evaluation. All recipes
use the same vLLM evaluation harness, regardless of which framework trained the model.

## How the adapter works

| Responsibility | Extension | Implementation |
|---|---|---|
| Run an episode | `--custom-agent-function-path`, inline mode | [`agent.py`](rft_miles/agent.py) |
| Keep replayed turns in one conversation branch | `--session-message-matcher` | [`session.py`](rft_miles/session.py) |
| Supply the reward | `--custom-rm-path` | [`reward.py`](rft_miles/reward.py) |
| Check which tokens survive collection | `--session-sample-postprocessor-path` | [`attest.py`](rft_miles/attest.py) |
| Publish a saved checkpoint | `--custom-megatron-post-save-hook-path` | [`checkpoint.py`](rft_miles/checkpoint.py) |
| Resolve and launch the recipe | Model launcher plus `--extra-args` | [`render.py`](rft_miles/render.py), [`launch.py`](rft_miles/launch.py) |

An invalid or infrastructure-failed episode has no reward. miles drops its GRPO group rather
than treating the failure as an incorrect answer. Repeated exclusions stop the run; a judge
outage first pauses scoring for up to `train.judge_outage_wait_s`.

The cookbook's [Qwen launcher](rft_miles/launchers/qwen3_6_35b_a3b.py) matches the verl recipe's
optimizer, loss, parallelism, and training budget. It disables MTP training, speculative decoding,
and **MoE load-balancing loss**. Keep that loss disabled when reproducing the comparison.
Upstream launchers are used for the GLM configurations.

## Known limitations

- **Training shutdown can be reported as a failure.** A checkpoint may be published and node
  scripts may exit successfully while AI Runtime marks the job failed during teardown. Verify
  the checkpoint and logs before retrying. A no-op retry is not a clean training smoke test.
- **Async rollouts use older weights.** The next rollout is generated while the previous one
  trains, so its policy is one update behind. This differs from verl's configurable lag.
- **A retry can create another session branch.** Each branch becomes a sample. The message
  matcher keeps ordinary turns together, but cannot make a changed request history identical.
- **Attestations are separate files.** miles builds samples after the episode finishes, so the
  attestations live under `<episode dir>/attestations`, linked by episode ID.
- **Startup needs the pinned source archive.** An unavailable archive or a hash mismatch stops
  the job before training.

## Testing

For adapter tests without installing miles:

```bash
pip install -e '.[dev,eval]' --no-deps
pip install pytest pyyaml regex sympy pylatexenc transformers openai
pip install --no-deps -e frameworks/miles
pytest frameworks/miles/tests
```

The framework-API tests also need CPU PyTorch and the pinned miles source. The `miles-api` job
in [the CI workflow](../../.github/workflows/ci.yml) provides the exact setup and verifies the
source hash. These CPU tests check hook signatures, flags, and sample handling; use a GPU smoke
test to check the full training environment.
