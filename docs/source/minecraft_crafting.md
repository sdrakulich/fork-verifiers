# Minecraft Crafting Training Guide

This guide walks through the `minecraft_crafting.py` example located in `verifiers/examples/`. The script demonstrates how the main abstractions of the Verifiers framework work together to train a model that answers crafting questions using data pulled from the Minecraft wiki.

## Overview

1. **Dataset** – The script collects recipes by calling the public MediaWiki API. Each entry asks “How do you craft X?” and stores the recipe as the answer.
2. **Parser** – A basic `Parser` instance extracts the model’s final answer from `<answer>` tags.
3. **Rubric** – Rewards are calculated from simple string matching (`recipe_reward`) plus a formatting reward from `parser.get_format_reward_func()`.
4. **Environment** – `SingleTurnEnv` orchestrates prompts, responses and rewards for one-shot Q&A.
5. **GRPOTrainer** – Runs Group Relative Policy Optimization using the environment’s rewards.

## Running the Example

Execute the script directly to fetch recipes and start training:

```bash
python -m verifiers.examples.minecraft_crafting
```

The example defines a small list of items in the `ITEMS` constant. Adjust this list to train on additional recipes.

## Step-by-Step Breakdown

```python
import verifiers as vf
from verifiers.examples.minecraft_crafting import build_dataset, parser, rubric, system_prompt

# Items for the dataset
items = ["Wooden Pickaxe", "Stone Pickaxe", "Cake"]

# Create the dataset from the wiki
dataset = build_dataset(items)

# Create an environment using the same parser and rubric as the example
env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    message_type="chat",
)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=vf.grpo_defaults(run_name="minecraft-crafting"),
)
trainer.train()
```

The trainer pulls rewards from the environment and optimizes the policy with GRPO. The resulting model should answer “How do you craft X?” with accurate recipes sourced directly from the wiki.
