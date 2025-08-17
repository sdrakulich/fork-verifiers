# Task efficacy environment package.
#
# This package exposes load_environment to construct a multi-turn
# environment for evaluating a candidate documentation against a
# taskset derived from a gold standard. Each row in the dataset
# supplies a single task definition under the `info['task']` key and a
# candidate/gold document pair. The environment drives multiple
# turns with a language model to attempt the task and automatically
# checks the model's responses via deterministic checkers defined in checkers.

from .env import load_environment  # noqa: F401

__all__ = ["load_environment"]
