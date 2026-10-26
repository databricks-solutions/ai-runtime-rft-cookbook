# OfficeQA miles smoke: shutdown caveat

This run exercised the recipe runner with `databricks air` and images from Unity Catalog:
two training updates, a restart and resumed third update, export, evaluation and comparison.
It used the measured verl run's baseline. The qualification record is `smoke-tested`, with
all seven checks passing, and the export hashes the content of the 33 weight shards.

**The training jobs did not terminate cleanly.** Both attempts that actually trained published
their checkpoints and finished their node scripts with exit code 0, but AI Runtime marked the
jobs failed. Retrying each stage found its target update already published, did no further
training and succeeded. Export, evaluation and comparison then succeeded.

The failed training logs show SGLang's detokenizer exiting with SIGTERM during shutdown,
followed by SIGQUIT and crash diagnostics. That is a lead, not a confirmed explanation for
the platform's failed status. A separate orphan-reaping experiment did not resolve it and
was reverted.

This record demonstrates durable checkpoints, restart/resume, export and complete evaluation.
It must not be cited as evidence that both miles training jobs shut down successfully. A clean
training smoke remains open. The original qualification and receipts are preserved as written;
this note does not change their results.
