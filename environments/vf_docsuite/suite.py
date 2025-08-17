import verifiers as vf


def load_environment(**kwargs):
    """
    Load an environment group for the docpair evaluation suite. This combines the
    structural coverage, task efficacy, and editorial actionability environments.

    Parameters:
        structure_env: preloaded environment instance for structure metrics.
        task_env: preloaded environment instance for task-based evaluation.
        editorial_env: preloaded environment instance for editorial evaluation.

    Returns:
        vf.EnvGroup: A group combining the provided environments with names for
        each axis (coverage, task, editorial).
    """
    # Extract preloaded environments from kwargs. These should be instances of
    # verifiers.Env or its subclasses. They are expected to be loaded before
    # constructing the suite.
    struct_env = kwargs.get("structure_env")
    task_env = kwargs.get("task_env")
    edit_env = kwargs.get("editorial_env")

    # Create the environment group. The env_names parameter assigns labels to
    # each environment, which will appear in evaluation reports.
    return vf.EnvGroup(
        envs=[struct_env, task_env, edit_env],
        env_names=["coverage", "task", "editorial"]
    )
