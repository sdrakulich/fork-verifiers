# Learning Verifiers with Minecraft Crafting

This walkthrough demonstrates the entire verifiers framework through a practical reinforcement learning example. The goal is to train a model to correctly answer questions such as "How do you craft a cake?" using recipes automatically pulled from the [Minecraft Wiki](https://minecraft.wiki).

The full script lives in `verifiers/examples/minecraft_crafting_walkthrough.py` and is heavily commented for a top‑to‑bottom explanation. This document complements those comments with broader context about each abstraction.

## 1. Collecting a Dataset

We use the public MediaWiki API to fetch wikitext for each item. A helper function `build_dataset` parses the crafting grid templates and converts them to plain text. The resulting list of rows is turned into a HuggingFace `Dataset`:

```python
rows = []
for item in items:
    wt = fetch_wikitext(item)
    recipe = parse_crafting_grid(wt)
    if recipe:
        rows.append({"question": f"How do you craft {item}?", "answer": recipe})

dataset = Dataset.from_list(rows)
```

## 2. Parsing Model Output

`Parser` objects define how to extract fields like `<answer>` from the model's response. The walkthrough uses the base `Parser`, which looks for tags directly. Any parser from `verifiers.parsers` can be swapped in.

```python
parser = vf.Parser()
```

## 3. Designing a Rubric

Rewards are computed by a `Rubric`, which combines multiple functions. We reward exact recipe mention and well‑formed XML tags:

```python
def recipe_reward(completion, answer, **_):
    text = parser.parse_answer(completion) or ""
    return 1.0 if answer.lower() in text.lower() else 0.0

rubric = vf.Rubric(
    funcs=[recipe_reward, parser.get_format_reward_func()],
    weights=[1.0, 0.2],
)
```

Rubrics can aggregate any number of custom reward functions, enabling complex evaluations.

## 4. Creating an Environment

Environments orchestrate prompts, responses and rewards. `SingleTurnEnv` is ideal for one‑shot Q&A. The walkthrough supplies the dataset, the parser and the rubric:

```python
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    message_type="chat",
)
```

## 5. Training with GRPO

Group Relative Policy Optimization (GRPO) is the RL algorithm used across the examples. Training looks like this:

```python
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
training_args = vf.grpo_defaults(run_name="minecraft-walkthrough")

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
```

During training, the environment produces rewards for each rollout and the trainer updates the policy accordingly.

## 6. Running the Example

1. Install dependencies (datasets, requests, openai, pytest-asyncio).
2. Run the script directly:

```bash
python -m verifiers.examples.minecraft_crafting_walkthrough
```

The model learns to produce crafting recipes extracted straight from the wiki. You can expand `ITEMS` with any page title from the wiki to build a larger dataset.

## 7. Where to Go Next

The walkthrough highlights these abstractions:

- **Dataset** – Data containers from the `datasets` library.
- **Parser** – Extracts structured answers from model output.
- **Rubric** – Combines reward signals.
- **Environment** – Manages prompting and scoring.
- **GRPOTrainer** – Optimizes the model with reinforcement learning.

Explore the rest of the `verifiers` package for more parsers, environment types, and rubric utilities. The other example scripts in `verifiers/examples/` show additional patterns such as multi‑environment training, tool execution, and self‑reward setups.

