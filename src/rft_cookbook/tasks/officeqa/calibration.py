# Derived from verl-on-air (Apache License 2.0), usecases/officeqa/judge_selfcheck.py at b088a42;
# modified for ai-runtime-rft-cookbook: cases are episode records driven through the harness. The
# cases are unchanged. See THIRD_PARTY_NOTICES.md.
"""Calibrate the support judge before it scores anything: five correct answers, two with
supported evidence paths and three without, each graded twice.

    python -m rft_cookbook.tasks.officeqa.calibration --resources resources.json --out calibration.json

Every grade must match and every call must return a verdict; otherwise the judge does not pass
and the stage that wanted it stops.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

from rft_cookbook.agent.episode import EpisodeRecord
from rft_cookbook.agent.harness import ByteCounter, ChatTurn, run_episode
from rft_cookbook.agent.policy import ParsedCall
from rft_cookbook.scoring.base import score_episode
from rft_cookbook.tasks import officeqa
from rft_cookbook.tools.base import FunctionTool, ToolSet

DEFENSE = "What were U.S. national defense expenditures in fiscal year 1940, in millions of dollars?"
INCREASE = "How much did national defense expenditures increase from FY1940 to FY1941, in millions of dollars?"
LOOKUP = {"id": "a", "observation": "obs_1", "claim": "FY1940 national defense expenditures were 2,602 million dollars."}


class _Script:
    def __init__(self, turns: list[ChatTurn]):
        self.turns = list(turns)

    async def complete(self, messages, tools, max_tokens):
        return self.turns.pop(0)


async def _episode(name: str, question: str, observations: list[tuple[str, dict, str]], report: dict) -> EpisodeRecord:
    """An episode that makes the given tool calls (each returns its fixed text), then submits `report`."""
    schemas = {s["function"]["name"]: s for s in officeqa.tool_schemas()}
    texts = {tool: text for tool, _, text in observations}
    tools = ToolSet([FunctionTool(lambda _t=tool, **_: texts[_t], schemas[tool]) for tool in dict.fromkeys(texts)])
    task = dataclasses.replace(officeqa.task(), tools=tools, scorer=None)
    turns = [ChatTurn(text="", tool_calls=[ParsedCall(tool, json.dumps(arguments), f"c{i}")])
             for i, (tool, arguments, _) in enumerate(observations)]
    turns.append(ChatTurn(text="", tool_calls=[ParsedCall(officeqa.TERMINAL_TOOL, json.dumps(report), "submit")]))
    row = {"id": f"calibration-{name}", "split": "calibration", "messages": [{"role": "user", "content": question}],
           "ground_truth": "", "metadata": {"question": question}}
    return await run_episode(task, _Script(turns), row, counter=ByteCounter(), episode_id=f"calibration-{name}")


async def cases() -> list[tuple[str, EpisodeRecord, str, float, str]]:
    """(name, episode, gold answer, expected score, question)."""
    read = ("read_document", {"file_name": "treasury_bulletin_1941_06.txt"})
    out = []
    for name, text, expected in [
        ("supported_lookup", "Treasury Bulletin 1941-06\nExpenditures (millions of dollars)\nNational defense | FY1940 | 2,602", 1.0),
        ("wrong_period", "Treasury Bulletin 1942-06\nExpenditures (millions of dollars)\nNational defense | FY1941 | 2,602", 0.0),
        ("wrong_category", "Treasury Bulletin 1941-06\nExpenditures (millions of dollars)\nAgriculture | FY1940 | 2,602\n"
                           "National defense | FY1940 | 6,404", 0.0),
    ]:
        out.append((name, await _episode(name, DEFENSE, [(*read, text)], {"answer": "2,602", "path": [LOOKUP]}), "2602",
                    expected, DEFENSE))
    out.append(("ungrounded_compute", await _episode(
        "ungrounded_compute", DEFENSE, [("compute", {"code": "print(2602)"}, "Output:\n2602")],
        {"answer": "2602", "path": [{**LOOKUP, "claim": "Computed FY1940 national defense expenditures: 2602."}]}),
        "2602", 0.0, DEFENSE))
    out.append(("supported_arithmetic", await _episode(
        "supported_arithmetic", INCREASE,
        [("read_document", {"file_name": "treasury_bulletin_1942_06.txt"},
          "Treasury Bulletin 1942-06\nNational defense expenditures (millions of dollars) | FY1940 2,602 | FY1941 6,404"),
         ("compute", {"code": "print(6404-2602)"}, "Output:\n3802")],
        {"answer": "3,802", "path": [
            {"id": "a", "observation": "obs_1", "claim": "National defense was 2,602 million in FY1940 and 6,404 million in FY1941."},
            {"id": "b", "observation": "obs_2", "claim": "Increase is 6404 minus 2602 = 3802 million dollars.",
             "depends_on": ["a"]}]}),
        "3802", 1.0, INCREASE))
    return out


async def calibrate(scorer, *, repeats: int = 2) -> dict[str, Any]:
    checks = []

    async def check(name, episode, gold, expected, question, repeat):
        record = EpisodeRecord.from_json(episode.to_json())   # each grade on its own copy
        score = await score_episode(scorer, record, gold, {"question": question})
        called = bool(score.components.get("judge_called"))
        checks.append({"case": name, "repeat": repeat, "expected": expected, "score": score.value,
                       "valid": score.valid, "judge_called": called, "detail": score.detail,
                       "judge": (score.judge or {}).get("model"),
                       "pass": score.valid and called and score.value == expected})

    built = await cases()
    try:
        await asyncio.gather(*(check(*case, repeat) for repeat in range(repeats) for case in built))
    finally:
        close = getattr(getattr(scorer, "judge", None), "aclose", None)
        if close:
            await close()
    checks.sort(key=lambda c: (c["case"], c["repeat"]))
    return {"status": "PASS" if checks and all(c["pass"] for c in checks) else "FAIL", "policy": scorer.policy,
            "checks": checks}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    resources = json.loads(args.resources.read_text())
    result = asyncio.run(calibrate(officeqa.build_scorer(resources["judge"])))
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
