# Derived from verl-on-air (Apache License 2.0), usecases/math/tests/test_grading.py at 8816f63;
# modified for ai-runtime-rft-cookbook: tests the vendored grader. See THIRD_PARTY_NOTICES.md.
"""Math grading: one extractor, one grader, a labelled edge-case suite.

Every label is the mathematically right answer; the cases the PRM800K grader gets wrong are strict
xfails, documenting its limits.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rft_cookbook.tasks.math import grading as g

EQUIVALENT = [
    ("0.75", "\\frac{3}{4}", "a decimal and its fraction (the reviewer's case)"),
    ("2\\sqrt{2}", "\\sqrt{8}", "a simplified radical (the reviewer's case)"),
    ("\\frac{1}{2}", "0.5", "fraction and decimal, other way round"),
    ("1/2", "\\frac{1}{2}", "a slash fraction"),
    ("\\dfrac{3}{4}", "\\frac{3}{4}", "\\dfrac"),
    ("\\tfrac{3}{4}", "\\frac{3}{4}", "\\tfrac"),
    ("-\\frac{1}{3}", "\\frac{-1}{3}", "where the minus sign sits"),
    ("\\frac{\\sqrt{2}}{2}", "\\frac{\\sqrt2}{2}", "an unbraced \\sqrt argument"),
    ("\\sqrt{2}/2", "\\frac{\\sqrt{2}}{2}", "a radical over a slash"),
    ("x^2+1", "1+x^2", "reordered terms"),
    ("5.0", "5", "a float that is an integer"),
    ("1,000", "1000", "a thousands separator"),
    ("-2", "- 2", "spacing after a minus"),
    ("10", "10\\%", "a percent sign"),
    ("\\$5", "5", "a dollar sign"),
    ("90^\\circ", "90", "a degree sign"),
    ("x=3", "3", "a variable assignment"),
    ("(1,2)", "(1, 2)", "tuple spacing"),
    ("\\left(1,2\\right)", "(1,2)", "\\left( \\right)"),
    ("(3, \\frac{\\pi}{2})", "\\left( 3, \\frac{\\pi}{2} \\right)", "a polar point"),
    ("[1,2)", "[1, 2)", "a half-open interval"),
    ("3\\pi", "3 \\pi", "a multiple of pi"),
    ("\\infty", "\\infty", "infinity"),
    ("\\text{(C)}", "\\text{(C)}", "a multiple-choice letter"),
    ("(C)", "\\text{(C)}", "a letter without \\text"),
    ("\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}", "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}", "a matrix"),
]
DIFFERENT = [
    ("0", "0.00001", "no absolute tolerance (the reviewer's case)"),
    ("\\frac{3}{4}", "0.7500001", "no relative tolerance either"),
    ("50", "0.5", "no percent rescaling"),
    ("\\frac{2}{4}", "\\frac{1}{2}", "MATH answers are in lowest terms"),
    ("\\sqrt{2}", "1.41421", "a decimal approximation of an irrational"),
    ("3.14", "\\pi", "pi is not 3.14"),
    ("12", "12.5", "an integer against a decimal"),
    ("3", "4", "a plain wrong number"),
    ("5", "-5", "the sign"),
    ("x^2", "x^3", "the exponent"),
    ("(1,2)", "(2,1)", "point coordinates are ordered"),
    ("(1,2)", "(1,2,3)", "the arity"),
    ("[1,2)", "(1,2)", "an interval's closed end"),
    ("-\\infty", "\\infty", "the sign of infinity"),
    ("", "5", "no answer"),
]
PRIME_MATH_MISSES = [   # equivalent, but prime_math refuses or fails them
    ("e^{2}", "e^2", "prime_math never gives sympy a braced exponent '^{' (its hang guard)"),
    ("10^{3}", "1000", "the same guard"),
    ("1000000", "10^6", "an integer answer against a non-integer-looking gold is refused"),
]


@pytest.mark.parametrize("pred,gold,why", EQUIVALENT, ids=[w for *_, w in EQUIVALENT])
def test_equivalent_answers_are_correct(pred, gold, why):
    assert g.grade(pred, gold) is True


@pytest.mark.parametrize("pred,gold,why", DIFFERENT, ids=[w for *_, w in DIFFERENT])
def test_different_answers_are_not(pred, gold, why):
    assert g.grade(pred, gold) is False


@pytest.mark.xfail(strict=True, reason="known prime_math limitation; see PRIME_MATH_MISSES")
@pytest.mark.parametrize("pred,gold,why", PRIME_MATH_MISSES, ids=[w for *_, w in PRIME_MATH_MISSES])
def test_known_prime_math_limits(pred, gold, why):
    assert g.grade(pred, gold) is True


def test_no_answer_and_an_overlong_answer_are_never_correct():
    assert g.grade(None, "5") is False and g.grade("5", None) is False
    assert g.grade("5" + " " * g.MAX_PRED_CHARS + "5", "55") is False


def test_model_output_is_never_executed(tmp_path: Path):
    canary = tmp_path / "pwned"
    code = f"__import__('pathlib').Path('{canary}').touch()"
    for gold in ("5", "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}"):   # the matrix path, eval()'d by prime_math.math_equal (not vendored)
        assert g.grade(code, gold) is False and g.grade(f"[{code}]", gold) is False
    assert not canary.exists()


@pytest.mark.parametrize("text,want", [
    ("#### 1\nwait, let me recheck: \\boxed{2}", "2"),          # the reviewer's case: the later box wins
    ("\\boxed{2} ... on reflection \\boxed{3}", "3"),
    ("so \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}"),                  # braces balanced
    ("\\boxed {5}", "5"),
    ("\\fbox{4}", "4"),
    ("The total is #### 1,234", "1234"),
    ("the answer is 7", None),                                     # never a bare number
    ("\\boxed{unclosed", None),
    ("", None),
])
def test_the_final_answer_is_the_last_box_else_an_explicit_hash(text, want):
    assert g.extract_answer(text) == want
