import verifiers as vf
from .rubric import rule_editorial, judge_editorial


def load_environment(**kwargs):
    parser = vf.ThinkParser()
    rule = vf.Rubric(funcs=[rule_editorial], weights=[1.0])
    judge = judge_editorial()
    combined = vf.RubricGroup([rule, judge])  # raw metrics kept even if weights later change
    return.SingleTurnEnv(
        dataset=kwargs["dataset"],
        parser=parser,
        rubric=combined,
        message_type=kwargs.get("message_type","chat"),
    )
