# Check SGLang shutdown

Use this probe to test one SGLang server without running training or Ray. It requires the miles
image and a small model already staged on your volume, such as the math recipe's Qwen3.5-2B.
Each job uses one H100, has a 30-minute limit, and does not change checkpoints.

From the repository root, with your workspace settings in `site.env`:

```bash
source site.env
databricks air run -f frameworks/miles/diagnostics/sglang_stop.yaml -p "$RFT_PROFILE" \
  --override environment.unity_catalog_image="$RFT_MILES_IMAGE" \
  --override env_variables.MODE=clean \
  --override env_variables.MODEL="$RFT_VOLUME_ROOT/models/Qwen3.5-2B@15852e8c1636"
```

Start with `clean`, then change `MODE` to compare shutdown paths:

| Mode | Signal target | Diagnostics |
|---|---|---|
| `clean` | Server process | Default settings |
| `crash` | Detokenizer | Default settings |
| `nodiag` | Detokenizer | Python stack and CUDA dump diagnostics disabled |
| `group` | Server's process group | Default settings; closer to miles' teardown |

The probe requires a healthy server and a successful generation before sending the signal. It
runs the server in its own process group, cleans up only that group, and waits for the server.
The stop allowance defaults to 180 seconds: diagnostic stack collection and the CUDA-dump wait
can exceed a minute. Override `env_variables.STOP_WAIT_S` if necessary.

Read the child exit code, final probe message, and platform job status together. A server may
exit with code 137 after handling a fatal signal while the probe itself successfully exits 0.
A setup failure or stop timeout makes the probe fail. Those are different outcomes.

This isolates engine shutdown only. Passing does not establish that a full miles training job,
with Ray actors, optimizer state, and distributed communication, shuts down correctly. A SIGQUIT
or crash-diagnostic message alone does not establish a platform failure either.

The CPU tests use a fake server and need no GPU or SGLang installation:

```bash
pytest frameworks/miles/tests/test_sglang_stop.py
```

They check all four modes, startup failure, and that similarly named unrelated processes survive.
