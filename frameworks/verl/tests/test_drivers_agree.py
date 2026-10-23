"""One scripted episode, two drivers: verl's native loop and the Chat Completions harness make the
same decisions and earn the same score, because both call the shared policy and scorer."""
from __future__ import annotations

import asyncio

from test_agent_loop import ScriptedServer, calculator_call, episodes, make_loop, run

from rft_cookbook.agent.harness import ChatTurn, TemplateCounter, run_episode
from rft_cookbook.agent.policy import ParsedCall
from rft_cookbook.scoring.base import score_episode
from rft_cookbook.tasks import load_task


def decisions(record):
    return {
        "termination": record.termination.reason, "answer": record.answer, "turns": len(record.turns),
        "calls": [(c.name, c.status, c.delivery, c.delivered_text, c.delivered_to_turn) for c in record.calls()],
        "texts": [t.text.strip() for t in record.turns], "reasoning": [t.reasoning for t in record.turns],
        "score": (record.score.value, record.score.valid, record.score.policy),
    }


def test_native_loop_and_harness_agree(tokenizer, tokenizer_dir, tmp_path):
    first, final = calculator_call("12*7+3"), "Done.\n</think>\n\n\\boxed{87}"
    run(make_loop(tokenizer, tokenizer_dir, ScriptedServer(tokenizer, [first, final]), tmp_path), ground_truth="87")
    (native,) = episodes(tmp_path)

    task = load_task("math", max_response_tokens=1024)
    client_turns = [
        ChatTurn(text="I will compute it.\n</think>\n\n",
                 tool_calls=[ParsedCall("calculator", '{"expression": "12*7+3"}', "c0")]),
        ChatTurn(text=final),
    ]

    class Client:
        async def complete(self, messages, tools, max_tokens):
            return client_turns.pop(0)

    row = {"id": "q1", "split": "train", "messages": [{"role": "user", "content": "What is 2+2?"}],
           "ground_truth": "87", "metadata": {}}
    harness = asyncio.run(run_episode(task, Client(), row, counter=TemplateCounter(tokenizer)))
    asyncio.run(score_episode(task.scorer, harness, "87"))
    assert decisions(native) == decisions(harness)
    assert native.score.value == 1.0
    # both drivers measure the same context: the harness counts with the chat template
    assert harness.turns[0].prompt_tokens == native.turns[0].prompt_tokens > 100
