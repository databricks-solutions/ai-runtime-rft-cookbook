"""Convert canonical task rows to the parquet verl's RLHFDataset reads.

    python -m rft_verl.data --task math --rows <dir>/train.jsonl --out <dir>/verl/train.parquet

Each row becomes: `prompt` (the task's system prompt, then the row's messages), `agent_name`
(the cookbook agent loop), `data_source` (the task), `reward_model.ground_truth`, and
`extra_info` with the row id, split, the system prompt's hash and the row metadata as JSON (so
the parquet schema does not depend on which metadata keys a row has).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rft_cookbook.agent.harness import initial_messages
from rft_cookbook.agent.spec import sha256_text
from rft_cookbook.data import file_sha256, read_rows
from rft_cookbook.tasks import load_task
from rft_verl.agent_loop import AGENT_NAME


def to_verl_rows(task, rows):
    prompt_sha = sha256_text(task.system_prompt)
    for row in rows:
        yield {
            "data_source": task.name,
            "agent_name": AGENT_NAME,
            "prompt": initial_messages(task, row),
            "ability": task.name,
            "reward_model": {"style": "rule", "ground_truth": json.dumps(row["ground_truth"])
                             if not isinstance(row["ground_truth"], str) else row["ground_truth"]},
            # the ground truth as the task wrote it (a list of golds stays a list); reward_model holds text
            "extra_info": {"id": str(row["id"]), "split": row["split"] or "", "prompt_sha256": prompt_sha,
                           "metadata_json": json.dumps(row.get("metadata") or {}, sort_keys=True),
                           "ground_truth_json": json.dumps(row["ground_truth"])},
        }


def convert(task_name: str, rows_path: Path, out: Path, budgets: dict | None = None) -> dict:
    """`budgets`: the recipe's agent budgets, which the system prompt can name (OfficeQA's turns)."""
    import pandas as pd

    task = load_task(task_name, **(budgets or {}))
    frame = pd.DataFrame(list(to_verl_rows(task, read_rows(rows_path))))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    return {"task": task.name, "task_version": task.version, "source": str(rows_path),
            "source_sha256": file_sha256(rows_path), "rows": len(frame), "parquet": str(out),
            "prompt_sha256": sha256_text(task.system_prompt)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    record = convert(args.task, args.rows, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
