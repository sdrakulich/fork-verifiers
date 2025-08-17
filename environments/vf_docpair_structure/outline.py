"""Utilities for extracting and normalising document outlines.

The functions in this module perform lightweight parsing of Markdown
text to extract heading lines and canonicalise them for comparison.
Headings are identified by leading ``#`` characters (from level 1 to
level 6).  The returned outline is a list of strings prefixed by the
heading level (e.g. ``"2:usage"`` for a level‑2 "Usage" section).
"""

from __future__ import annotations

import re
from typing import List

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def extract_outline(md_text: str) -> List[str]:
    """Extract a list of headings from a Markdown document.

    Parameters
    ----------
    md_text : str
        The raw Markdown text.

    Returns
    -------
    List[str]
        A list of headings prefixed with their depth (e.g.
        ``"3:installation"``).  Non‑heading lines are ignored.
    """
    headings: List[str] = []
    for line in md_text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            level = len(match.group(1))
            title = re.sub(r"\s*#+\s*$", "", match.group(2)).strip()
            headings.append(f"{level}:{title}")
    return headings


def canonicalize_outline(outline: List[str]) -> List[str]:
    """Normalise a list of heading strings for comparison.

    This function lowercases all characters, removes non‑alphanumeric
    symbols (except colons, hyphens and spaces), and collapses
    consecutive whitespace.  It does not alter the numeric prefix.
    """
    def normalise(s: str) -> str:
        # split numeric prefix from the actual title
        parts = s.split(":", 1)
        if len(parts) != 2:
            return s.lower()
        level, title = parts
        title = title.lower()
        title = re.sub(r"[^a-z0-9:/\- ]+", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        return f"{level}:{title}"

    return [normalise(h) for h in outline]