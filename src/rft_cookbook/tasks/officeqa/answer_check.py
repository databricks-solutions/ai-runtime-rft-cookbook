# Derived from verl-on-air (Apache License 2.0), usecases/officeqa/answer_check.py at 495b583;
# modified for ai-runtime-rft-cookbook: package-relative import; dash and accounting negatives; text
# golds matched as the whole answer. See THIRD_PARTY_NOTICES.md.
"""OfficeQA answer check: the submitted answer against gold, with scale, sign and list length.

The vendored matcher (``answer_match``) compares bare numbers, so it accepts "543 billion"
for "543 million" and rejects "543000000"; it reads "-$1,299 million" as a positive 1,299;
and it accepts any number in the answer that equals gold, extra list items included.
Numeric golds are compared here instead; a figure in parentheses, "(5)", is negative, as in the
Bulletin's tables, and so is one after an en dash. Golds that are only words ("Yes", a fund's
name) must be the answer, or lead it before its explanation: the vendored matcher accepts them
anywhere in it ("Yes" in "not yes"). Golds that mix words and figures (dates) keep the vendored
matcher, which checks both.

An answer must state its final value or values, alone. A single value must lead the
answer: everything from the first explanation marker on is working, and figures there do
not count. A list may carry parenthetical asides between its items, which do not count
either.
"""

import re
from dataclasses import dataclass

from rft_cookbook.tasks.officeqa.answer_match import has_significant_text, normalize_text, score_answer

WRONG = "wrong final answer"
NOT_LEADING = "the answer does not lead with the final value"
EXTRA = "the answer states more values than the final answer has"

_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
# A number that is not part of a word ("FY1984", "Q3"), its sign, and a unit written right after it.
_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>-?)(?P<digits>\d{1,3}(?:,\d{3})+|\d+)(?P<fraction>\.\d+)?"
    r"\s*(?P<unit>%|percent\b|(?:thousand|million|billion|trillion)s?\b)?",
    re.IGNORECASE,
)
# Where an answer's explanation starts: a parenthesis, a new line, a sentence end, a spaced
# dash, "=", ";", ":" or "i.e."
_EXPLANATION = re.compile(r"\(|\n|\.\s| [-–—] | = |;|:|\bi\.e\.")
# A figure in parentheses is an accounting negative, not an aside.
_PARENTHESIZED = r"\(\s*-?\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:%|percent|(?:thousand|million|billion|trillion)s?)?\s*\)"
_ASIDE = re.compile(rf"(?!{_PARENTHESIZED})\([^()]*\)")
_DASHES = str.maketrans({"\u2012": "-", "\u2013": "-", "\ufe63": "-", "\uff0d": "-"})
# Where a text answer's explanation starts, after the gold words.
_TEXT_END = re.compile(r"\s*(?:[,.;:(\[]|[-\u2013\u2014]\s)|\s+[-\u2013\u2014]")


@dataclass(frozen=True)
class Value:
    number: float
    unit: str | float | None  # "%", a scale multiplier, or None for a bare figure
    year: bool


def check(answer: str, gold: str, *, tolerance: float = 0.0) -> tuple[bool, str]:
    """(correct, reason); the reason is empty when the answer is correct."""
    gold_values = values(gold)
    if not gold_values:
        return (True, "") if _text_leads(answer, gold) else (False, WRONG)
    if has_significant_text(gold)[0]:
        return (True, "") if score_answer(gold, answer, tolerance=tolerance) > 0 else (False, WRONG)
    keep_years = any(value.year for value in gold_values)

    def countable(text: str) -> list[Value]:
        return [value for value in values(text) if keep_years or not value.year]

    stated = countable(leading_value(answer) if len(gold_values) == 1 else _ASIDE.sub(" ", answer))
    if _covers(gold_values, stated, tolerance):
        return (False, EXTRA) if len(stated) > len(gold_values) else (True, "")
    if _covers(gold_values, countable(answer), tolerance):
        return False, NOT_LEADING
    return False, WRONG


def values(text: str) -> list[Value]:
    """Every number in ``text`` with its sign and unit; currency signs are dropped."""
    found = []
    text = normalize_text(text).translate(_DASHES).replace("$", "") if text else ""
    for match in _NUMBER.finditer(text):
        number = float(match["digits"].replace(",", "") + (match["fraction"] or ""))
        unit = match["unit"]
        if unit is not None:
            unit = "%" if unit == "%" or unit.lower() == "percent" else _SCALES[unit.lower().rstrip("s")]
        year = unit is None and not match["fraction"] and "," not in match["digits"] and 1900 <= number <= 2100
        negative = match["sign"] or (re.search(r"\(\s*$", text[:match.start()])
                                     and re.match(r"\s*\)", text[match.end():]))
        found.append(Value(-number if negative else number, unit, year and not negative))
    return found


def _text_leads(answer: str, gold: str) -> bool:
    """A words-only gold is the whole answer, or its start up to where the explanation begins."""
    def clean(text: str) -> str:
        text = re.sub(r"\s+", " ", _ASIDE.sub(" ", text.lower())).strip().strip("\"'").strip()
        return re.sub(r"^the ", "", text).rstrip(".!")
    stated, expected = clean(answer), clean(gold)
    if not expected or not stated.startswith(expected):
        return False
    rest = stated[len(expected):]
    return not rest or _TEXT_END.match(rest) is not None


def leading_value(answer: str) -> str:
    """The answer up to where its explanation starts, once its first digit has appeared."""
    first_digit = re.search(r"\d", answer)
    cut = _EXPLANATION.search(answer, first_digit.end()) if first_digit else None
    return answer if cut is None else answer[: cut.start()]


def matches(gold: Value, stated: Value, tolerance: float = 0.0) -> bool:
    if "%" in (gold.unit, stated.unit):
        # A bare figure inherits the percent; a percent never equals an amount.
        return gold.unit in ("%", None) and stated.unit in ("%", None) and _same(gold.number, stated.number, tolerance)
    as_written = _same(gold.number, stated.number, tolerance)
    absolute = _same(gold.number * (gold.unit or 1), stated.number * (stated.unit or 1), tolerance)
    if gold.unit and stated.unit:
        return absolute  # "1.5 billion" is "1,500 million", but "543 billion" is not "543 million"
    # A bare figure is either in the other side's scale or the absolute amount.
    return as_written or absolute


def _covers(gold: list[Value], stated: list[Value], tolerance: float) -> bool:
    """Whether each gold value matches a different stated value."""
    unused = list(stated)
    for value in gold:
        hit = next((index for index, candidate in enumerate(unused) if matches(value, candidate, tolerance)), None)
        if hit is None:
            return False
        del unused[hit]
    return True


def _same(gold: float, stated: float, tolerance: float) -> bool:
    # A floor of 1e-9 absorbs float error from scaling, e.g. 1169.41 * 1e6.
    return gold == stated or abs(gold - stated) <= max(tolerance, 1e-9) * abs(gold)
