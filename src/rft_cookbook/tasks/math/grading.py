# Derived from verl-on-air (Apache License 2.0), usecases/math/grading.py at 8816f63;
# modified for ai-runtime-rft-cookbook: grades with the vendored PRM800K grader instead of
# importing verl. See THIRD_PARTY_NOTICES.md.
"""The one final-answer extractor and the one grader for math, used in training and evaluation.

extract_answer(text) -> the content of the LAST \\boxed{...} (or \\fbox{...}, brace-balanced),
    else the LAST explicit `#### <number>`, else None. Never a bare number: the last number in a
    truncated chain of thought is not an answer.

grade(pred, gold) -> PRM800K's grade_answer (as shipped in verl's prime_math): the MATH
    normalizer, then sympy on the normalized forms. EXACT after normalization and symbolic
    simplification: 0.75 == \\frac{3}{4}, 2\\sqrt{2} == \\sqrt{8}, (1,2) == (1, 2); but
    0 != 0.00001, 50 != 0.5, and \\frac{2}{4} != \\frac{1}{2} (MATH answers are in lowest terms).

Model output is untrusted. grade_answer refuses to hand expressions with more than two unknown
letters to sympy's parser; each sympy comparison runs in a forked child with a 10 s limit, and
predictions longer than MAX_PRED_CHARS are not graded. It never raises.
"""
from __future__ import annotations

import re
from typing import Any

MAX_PRED_CHARS = 400
_HASH_RE = re.compile(r"####\s*\$?(-?[\d,]*\.?\d+)")


def last_boxed(s: str) -> str | None:
    """Content of the LAST \\boxed{...}/\\fbox{...}, brace-balanced. None if absent or unclosed."""
    key = "\\boxed"
    i = s.rfind(key)
    if i < 0:
        key = "\\fbox"
        i = s.rfind(key)
        if i < 0:
            return None
    j = i + len(key)
    while j < len(s) and s[j] == " ":
        j += 1
    if j >= len(s) or s[j] != "{":
        return None
    depth, start = 0, j
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:j]
        j += 1
    return None


def extract_answer(text: str) -> str | None:
    boxed = last_boxed(text or "")
    if boxed is not None:
        return boxed.strip() or None
    hits = _HASH_RE.findall(text or "")
    return hits[-1].replace(",", "") if hits else None


def grade(pred: str | None, gold: Any) -> bool:
    if pred is None or gold is None:
        return False
    pred, gold = str(pred).strip(), str(gold).strip()
    if not pred or not gold or len(pred) > MAX_PRED_CHARS:
        return False
    from rft_cookbook.tasks.math._vendor.prm800k_grader import grade_answer

    try:
        return bool(grade_answer(pred, gold))
    except Exception:  # noqa: BLE001 - an answer that cannot be graded is not correct
        return False
