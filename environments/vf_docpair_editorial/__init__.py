# Editorial/clarity evaluation environment package.
#
# This package exposes load_environment to construct a single-turn
# environment for measuring the actionability and clarity of candidate
# documentation. It combines simple rule-based checks with a judge
# model rubric to evaluate how well a document communicates procedures,
# prerequisites and error handling.

from .env import load_environment  # noqa: F401

__all__ = ["load_environment"]
