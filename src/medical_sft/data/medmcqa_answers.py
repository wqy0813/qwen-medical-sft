"""The single strict parser for MedMCQA gold answers; no numeric-string coercion."""
from __future__ import annotations

from typing import Any

LETTERS = ("A", "B", "C", "D")


def parse_answer(value: Any = None) -> tuple[str, str | None]:
    """Return (valid/missing/invalid, letter), keeping missing distinct from invalid."""
    if value is None:
        return "missing", None
    if isinstance(value, bool):
        return "invalid", None
    if isinstance(value, int):
        return ("valid", LETTERS[value]) if value in (0, 1, 2, 3) else ("invalid", None)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "missing", None
        letter = stripped.upper()
        if letter in LETTERS:
            return "valid", letter
    return "invalid", None
