"""Math quickstart: competition math with a calculator, scored by the rule grader.

The answer is the last \\boxed{...} of the model's final turn (a turn without tool calls).
A correct answer scores 1, a wrong or missing one 0; the grader never fails, so every score
is valid.
"""
from __future__ import annotations

import asyncio
from typing import Any

from rft_cookbook.agent.spec import AgentTask, Budgets, TerminalRule
from rft_cookbook.scoring.base import RewardResult
from rft_cookbook.tasks.math.calculator import calculator_tool
from rft_cookbook.tasks.math.grading import extract_answer, grade
from rft_cookbook.tools.base import ToolSet

NAME = "math"
VERSION = "1"

SYSTEM_PROMPT = (
    "You are a careful competition-math problem solver. Reason step by step. "
    "Whenever you need to do arithmetic, call the `calculator` tool with a single "
    "arithmetic expression (e.g. {\"expression\": \"12 * 7 + 3\"}) instead of "
    "computing it in your head, and use its result. You may call the tool several "
    "times. When you are confident, stop calling tools and give your final answer "
    "on its own line in the exact form \\boxed{<answer>} (put ONLY the final answer "
    "inside the box, e.g. \\boxed{\\frac{1}{2}} or \\boxed{24})."
)

BUDGETS = Budgets(max_turns=4, max_response_tokens=4096, max_turn_tokens=2048, generation_reserve=128,
                  max_tool_calls_per_turn=1, tool_output_bytes=1024, tool_timeout_s=10.0)


class MathRuleScorer:
    policy = "math/rule-v1"
    components = ("answered", "correct")

    async def score(self, episode, ground_truth: Any, metadata: dict[str, Any]) -> RewardResult:
        answer = episode.answer
        if answer is None:
            return RewardResult(valid=True, score=0.0, components={"answered": 0.0, "correct": 0.0})
        correct = await asyncio.get_running_loop().run_in_executor(None, grade, answer, ground_truth)
        return RewardResult(valid=True, score=float(correct), components={"answered": 1.0, "correct": float(correct)})


def task(**budget_overrides: Any) -> AgentTask:
    return AgentTask(
        name=NAME, version=VERSION, system_prompt=SYSTEM_PROMPT, tools=ToolSet([calculator_tool()]),
        terminal=TerminalRule.answer_pattern(extract_answer), budgets=BUDGETS.replace(**budget_overrides),
        scorer=MathRuleScorer())
