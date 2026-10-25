"""Check that every number in an agent answer comes from somewhere it may.

A number in an answer is grounded when it matches, at the precision it is
written with, a number from one of the sources: a tool result, a tool
argument, the user's question, or the fixed facts in the system prompt.
"92%" may quote 92.2 and "39.2K" may quote 39,157, but "39,000" may not.
Fractions also match as percentages (0.95 and "95%"). Signs are ignored,
because answers carry them in words ("18.5% more"). List markers at the start
of a line and the small counts 0, 1 and 2 are not checked.
"""

import json
import re
from dataclasses import dataclass

NUMBER = re.compile(
    r"(?<![\w.])\$?(?P<digits>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<decimals>\d+))?"
    r"(?P<suffix>[kKmM](?![A-Za-z]))?"
)
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
SMALL_COUNTS = {0.0, 1.0, 2.0}


@dataclass(frozen=True)
class Number:
    """A number as written: its text, value and the precision it is written with."""

    text: str
    value: float
    precision: float
    whole: bool


@dataclass(frozen=True)
class GroundingReport:
    grounded: bool
    checked: list[str]
    unverified: list[str]


def extract_numbers(text: str) -> list[Number]:
    """Numbers written in text, with K and M suffixes applied."""
    numbers = []
    for match in NUMBER.finditer(text):
        decimals = match.group("decimals") or ""
        value = float(
            match.group("digits").replace(",", "")
            + (f".{decimals}" if decimals else "")
        )
        precision = 10.0 ** -len(decimals)
        suffix = (match.group("suffix") or "").lower()
        scale = {"k": 1e3, "m": 1e6}.get(suffix, 1.0)
        numbers.append(
            Number(
                text=match.group(0),
                value=value * scale,
                precision=precision * scale,
                whole=not decimals and not suffix,
            )
        )
    return numbers


def matches(number: Number, value: float) -> bool:
    """True when value, rounded the way number is written, gives number."""
    tolerance = number.precision / 2 + 1e-9 * max(1.0, abs(number.value))
    return abs(abs(value) - number.value) <= tolerance


def source_values(sources) -> list[float]:
    """Every number in the sources, as itself and as a percentage."""
    values = []
    for source in sources:
        text = source if isinstance(source, str) else json.dumps(source)
        for number in extract_numbers(text):
            values.extend((number.value, number.value * 100))
    return values


def check_grounding(answer: str, sources) -> GroundingReport:
    """Report the numbers in answer that no source supports."""
    values = source_values(sources)
    checked, unverified = [], []
    for number in extract_numbers(LIST_MARKER.sub(" ", answer)):
        if number.whole and number.value in SMALL_COUNTS:
            continue
        checked.append(number.text)
        if not any(matches(number, value) for value in values):
            unverified.append(number.text)
    return GroundingReport(
        grounded=not unverified, checked=checked, unverified=unverified
    )


def mentions(answer: str, value: float) -> bool:
    """True when the answer writes value at some precision (used by the evaluations)."""
    return any(matches(number, value) for number in extract_numbers(answer))
