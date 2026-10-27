# OfficeQA miles smoke-test limitation

The checkpoint, resumed third update, export, and evaluation passed their qualification checks.
However, AI Runtime reported the jobs that performed training as failed during shutdown, even
though they had published their checkpoints. Successful retries had no training left to do.

This evidence supports checkpoint durability and evaluation, not reliable training-job shutdown.
The failure remains a known limitation of the miles workflow. The original qualification and
stage receipts are preserved without altering their results.
