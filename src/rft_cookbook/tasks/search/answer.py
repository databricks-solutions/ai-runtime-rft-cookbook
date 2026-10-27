# Derived from verl-on-air (Apache License 2.0), usecases/agentic-search/reward.py at 1c5653c;
# modified for ai-runtime-rft-cookbook: the answer is read from the model's final turn (the agent
# policy), so the role-span attribution and the retrieval bonus are gone. See THIRD_PARTY_NOTICES.md.
"""The answer of a search episode and its match against the gold answers.

The answer is the text of the last `<answer>...</answer>` block of the model's final turn (a turn
without tool calls). Answers and golds are normalized as in the SQuAD and HotpotQA evaluations
(lowercase, punctuation and articles dropped, whitespace collapsed); a question's golds are its
answer and its aliases.

    em        the normalized answer equals a normalized gold
    cover_em  a normalized gold occurs in the normalized answer
    f1        best token-overlap F1 against any gold
"""
from __future__ import annotations

import re
import string
from typing import Any

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)


def extract_answer(text: str) -> str | None:
    found = _ANSWER.findall(text or "")
    return found[-1].strip() if found else None


def normalize(text: str) -> str:
    text = "".join(ch for ch in str(text).lower() if ch not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


def golds(ground_truth: Any) -> list[str]:
    """The gold answers of a row: a list, or one string."""
    values = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth or [])
    return [str(v) for v in values if str(v).strip()]


def match(answer: str, ground_truth: Any) -> dict[str, float]:
    predicted = normalize(answer)
    normalized = [normalize(g) for g in golds(ground_truth)]
    return {"em": float(any(predicted == g for g in normalized)),
            "cover_em": float(any(g and g in predicted for g in normalized)),
            "f1": max((_f1(predicted, g) for g in normalized), default=0.0)}


def _f1(predicted: str, gold: str) -> float:
    ours, theirs = predicted.split(), gold.split()
    if not ours or not theirs:
        return float(ours == theirs)
    common = sum(min(ours.count(token), theirs.count(token)) for token in set(ours) & set(theirs))
    if not common:
        return 0.0
    precision, recall = common / len(ours), common / len(theirs)
    return 2 * precision * recall / (precision + recall)
