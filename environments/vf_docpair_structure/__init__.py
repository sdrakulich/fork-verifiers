"""Verifiers docpair structure environment package.

This module exposes a single function, :func:`load_environment`, which
constructs an instance of the structural evaluation environment.  The
environment computes structural similarity metrics between a candidate
documentation file and a reference (gold) file without invoking any
models.  It operates entirely algorithmically, extracting Markdown
headings from both documents, canonicalising them, and computing
recall- and depth-based scores.  See :mod:`.env` for the concrete
implementation.
"""

from .env import load_environment  # noqa: F401

__all__ = ["load_environment"]
