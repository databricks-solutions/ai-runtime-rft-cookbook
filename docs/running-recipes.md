# Running recipes

Use the recipe runner for a complete workflow, or submit individual stages when you need more
control. Both read the same checked-in recipes and jobs.

Before starting, build the appropriate [framework image](../frameworks/), upload it to your
workspace, and fill in `site.env` as described in the [quickstart](../README.md#quickstart).

## Choose a run name

```bash
python -m rft_cookbook.air.chain officeqa/verl/qwen3.6-35b-a3b_h100x16 officeqa-01
```

The run name selects its directory on your volume: `runs/officeqa-01/`. Use lowercase letters,
digits, dots, underscores, or hyphens, with 3–63 characters. Choose a new name for each
experiment; reuse a name only to continue the same run.

The runner saves job IDs under `~/.rft-runs/`. Restart an interrupted command with the same
recipe, name, workspace, and overrides. Successful stages are reused; active jobs are waited
for; failed stages are resubmitted. Do not change experiment settings midway through a run.

## Change settings

`--set` applies recipe overrides to every stage:

```bash
python -m rft_cookbook.air.chain officeqa/verl/qwen3.6-35b-a3b_h100x16 officeqa-seed7 \
  --set 'train.seed=7 eval.seed=42'
```

Keep the evaluation seed fixed when comparing training seeds. Other settings, such as learning
rate or agent budgets, also use dotted keys. Lists and scalar values follow YAML syntax.

A compatible run can reuse an existing base-model evaluation with
`eval.baseline_run=<run-name>`. The model and revision, questions, agent settings, judge, and
evaluation settings must match. Reuse saves evaluation time; it is not permission to compare
unlike experiments.

## Stages

| Stage | Purpose |
|---|---|
| `prep` | Prepare task data and the framework's input format |
| `stage_model` | Copy the pinned model to the volume; convert it for miles if needed |
| `stage_judge` | Stage the judge's tokenizer for OfficeQA |
| `calibrate` | Check the OfficeQA judge on known examples |
| `baseline_eval` | Evaluate the base model; this can run alongside training |
| `train` | Train and publish resumable checkpoints |
| `export` | Check the final Hugging Face weights |
| `eval` | Evaluate the exported checkpoint |
| `compare` | Compare scores by question and write the qualification record |

Training uses the recipe's full GPU allocation. Evaluation uses one GPU node. Preparation and
other lightweight stages use a small A10 node. A full run's duration includes startup, model
staging, checkpoint writes, and evaluation—not just optimizer updates.

## Submit one stage

Each recipe has a `jobs/` directory. Fill its workspace placeholders when submitting:

```bash
source site.env
databricks air run -f recipes/math/verl/qwen3.5-2b_h100x8/jobs/prep.yaml -p "$RFT_PROFILE" \
  --override env_variables.RFT_RUN_ID=math-01 \
  --override env_variables.RFT_VOLUME_ROOT="$RFT_VOLUME_ROOT" \
  --override environment.unity_catalog_image="$RFT_VERL_IMAGE" \
  --override secrets.HF_TOKEN="$RFT_HF_TOKEN_SECRET"
```

Use the same name and `env_variables.RFT_SET` overrides for every stage. Pass
`env_variables.RFT_CODE_COMMIT` and `env_variables.RFT_IMAGE` to record the code and image
identity. Search stages also need `env_variables.RFT_VS_INDEX` where the job requests it.
The recipe runner fills these fields for you.

Check a submitted job with:

```bash
databricks air get <job-id> -p "$RFT_PROFILE" -o json
databricks air logs <job-id> -p "$RFT_PROFILE" --download-to ./job-logs
```

## Resume training

Submit `train` again with a larger `train.total_training_steps` on verl, or `train.num_rollout`
on miles. The launcher loads the newest complete checkpoint. Keep the model, data, and other
training settings unchanged.

After training, export and evaluate the new checkpoint before comparing it with the baseline.
A resumed training job is not itself a new evaluation result.

## Check the result

On the volume, `comparison.json` contains the scores and confidence intervals.
`qualification.json` identifies the experiment and lists its checks. Inspect both: a successful
job submission or a checkpoint on disk is not enough to establish a valid comparison.

If evaluation is interrupted, retry with the same settings. The earlier incomplete attempt is
preserved under the evaluation directory's `attempts/`; its episodes are not mixed into the
retry. A completed evaluation or a changed evaluation plan requires a new label.

A failed qualification makes the comparison job fail, while preserving both result files for
inspection. The exported step must match the requested training budget, and the evaluated model
must identify this run's checkpoint and pinned base model.

For miles, a checkpoint may be complete even when AI Runtime reports a shutdown failure.
Confirm the published step and read the logs before retrying. A retry that finds nothing left
to train does not demonstrate that a training process shut down correctly. See the
[miles limitations](../frameworks/miles/README.md#known-limitations).

For a brief integration test rather than a full experiment, add `--smoke` to the runner command.
It trains two updates, restarts for a third, exports, and evaluates. Smoke-test scores should
not be presented as evidence that a full training recipe improves quality.
