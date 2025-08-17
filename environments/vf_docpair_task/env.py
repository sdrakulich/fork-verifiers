# Task efficacy evaluation environment.
#
# This environment orchestrates a multi-turn interaction between a
# language model and a synthetic user to test whether the model can
# complete tasks using only the candidate documentation. The tasks
# themselves are provided via the info['task'] dictionary in each
# dataset row. After each model reply the environment runs a
# deterministic checker to determine if the answer is correct. If the
# response is incorrect on the first attempt, a short hint is provided
# and the model is allowed one more attempt before the task is marked
# failed.

from __future__ import annotations

from typing import Dict, Any, List

from verifiers import ThinkParser, Rubric
from verifiers.envs import MultiTurnEnv

from .checkers import run_checker

# System prompt presented to the model for all tasks.
SYSTEM_PROMPT = (
    "You are evaluating documentation quality by performing tasks using "
    "ONLY the provided candidate doc. Answer succinctly and show commands "
    "exactly when requested. If information is missing, state what is missing "
    "explicitly."
)

# Mapping from task type to hint message.
HINTS = {
    "span": "Answer with the exact phrase from the document (no paraphrase).",
    "procedure": "List steps in order, using shell blocks with exact commands.",
    "rationale": "Provide a concise argument referencing specific doc sections.",
}


class DocpairTaskEnv(MultiTurnEnv):
    def __init__(self, dataset, rubric, max_turns: int = 3, **kwargs):
        super().__init__(dataset=dataset, rubric=rubric, max_turns=max_turns, **kwargs)

    def is_completed(self, messages: List[Dict[str, Any]], state: Dict[str, Any], **kwargs) -> bool:
        if state.get("success") or state.get("failed"):
            return True
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        return assistant_count >= self.max_turns

    def env_response(self, messages: List[Dict[str, Any]], state: Dict[str, Any], **kwargs):
        info: Dict[str, Any] = kwargs.get("info", {})
        task: Dict[str, Any] = info.get("task", {})
        if not messages:
            question = info.get("question", "")
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Candidate Documentation:\n\n{info.get('candidate_doc','')}\n\n"
                f"Task:\n{question}"
            )
            return {"role": "user", "content": prompt}, state

        last_assistant = messages[-1]["content"] if messages else ""
        if run_checker(last_assistant, task):
            state["success"] = True
            return {"role": "user", "content": "✅ Correct. Stop."}, state
        if not state.get("hinted"):
            state["hinted"] = True
            hint = HINTS.get(task.get("type"), "Answer more precisely.")
            return {"role": "user", "content": f"Hint: {hint}"}, state
        state["failed"] = True
        return {"role": "user", "content": "❌ Not correct. Stop."}, state


def task_reward(prompt, completion, answer, state, info, **_):
    return 1.0 if state.get("success") else 0.0


def load_environment(**kwargs) -> DocpairTaskEnv:
    parser = ThinkParser()
    rubric = Rubric(funcs=[task_reward], weights=[1.0])
    env = DocpairTaskEnv(dataset=kwargs.get("dataset"), rubric=rubric, max_turns=3, parser=parser)
    return env
