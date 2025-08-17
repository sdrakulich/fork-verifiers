# Environment for vf_docpair_task, handles multi-turn task completion
import verifiers as vf
from typing import Dict, Any
from .checkers import run_checker

# System prompt (not actively used here but kept for future extension)
SYSTEM = (
    "You are evaluating documentation quality by performing tasks using ONLY the provided CANDIDATE doc.\n"
    "Answer succinctly and show commands exactly when requested.\n"
    "If information is missing, state what is missing explicitly."
)

# Hints to give the model when initial attempt fails
HINTS = {
    "span": "Answer with the exact phrase from the document (no paraphrase).",
    "procedure": "List steps in order, using shell blocks with exact commands.",
    "rationale": "Provide a concise argument referencing specific doc sections."
}


class DocpairTaskEnv(vf.MultiTurnEnv):
    """Multi-turn environment for docpair task efficacy. Handles task checking and hinting."""

    def __init__(self, dataset, rubric, max_turns: int = 3, **kwargs):
        super().__init__(dataset=dataset, rubric=rubric, max_turns=max_turns, **kwargs)

    def is_completed(self, messages, state, **kwargs) -> bool:
        """Determine if the environment should terminate."""
        # Check for success or failure flags
        if state.get("success") is True:
            return True
        if state.get("failed") is True:
            return True
        # Count assistant responses; limit to max_turns
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        return len(assistant_msgs) >= self.max_turns

    def env_response(self, messages, state, **kwargs):
        """Process the last assistant message and respond accordingly."""
        info: Dict[str, Any] = kwargs.get("info", {})
        task: Dict[str, Any] = info.get("task", {})
        last = messages[-1]["content"] if messages else ""
        ok = run_checker(last, task)
        if ok:
            state["success"] = True
            return messages + [{"role": "user", "content": "✅ Correct. Stop."}], state
        # Provide one hint if not yet given
        if not state.get("hinted"):
            state["hinted"] = True
            hint = HINTS.get(task.get("type"), "Answer more precisely.")
            return messages + [{"role": "user", "content": f"Hint: {hint}"}], state
        # If second attempt fails, mark as failed
        state["failed"] = True
        return messages + [{"role": "user", "content": "❌ Not correct. Stop."}], state


def task_reward(prompt, completion, answer, state, info, **_) -> float:
    """Binary reward: 1.0 if task succeeded, else 0.0."""
    return 1.0 if state.get("success") else 0.0


def load_environment(**kwargs):
    """Factory function to instantiate the DocpairTaskEnv with appropriate rubric."""
    parser = vf.ThinkParser()
    rubric = vf.Rubric(funcs=[task_reward], weights=[1.0])
    return DocpairTaskEnv(dataset=kwargs["dataset"], rubric=rubric, max_turns=3, parser=parser)
