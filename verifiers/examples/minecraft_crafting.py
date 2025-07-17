"""Example RL training script for Minecraft crafting recipes using verifiers.

This script demonstrates the main abstractions of the verifiers library:

- **Dataset**: a HuggingFace ``Dataset`` containing question/answer pairs.
- **Parser**: extracts structured answers from model completions.
- **Rubric**: combines reward functions for evaluating rollouts.
- **Environment**: orchestrates prompts, model responses and rewards.
- **GRPOTrainer**: trains a model with Group Relative Policy Optimization.

The dataset is built programmatically from the public Minecraft wiki via the
MediaWiki API. Each entry asks how to craft an item and stores the wiki-derived
recipe as the answer. The environment is a ``SingleTurnEnv`` which issues the
question and expects the model to respond with the correct recipe.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import requests
from datasets import Dataset

import verifiers as vf

# ----------------------------------------------------------------------------
# Data collection utilities
# ----------------------------------------------------------------------------

API_URL = "https://minecraft.wiki/api.php"


def fetch_wikitext(page: str) -> str:
    """Retrieve raw wikitext for a wiki page."""
    resp = requests.get(
        API_URL,
        params={
            "action": "parse",
            "page": page,
            "prop": "wikitext",
            "format": "json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["parse"]["wikitext"]["*"]


def parse_crafting_grid(wikitext: str) -> str:
    """Extract a textual crafting recipe from ``Grid/Crafting Table`` templates."""
    match = re.search(r"\{\{[Gg]rid/Crafting Table(?P<content>.*?)\}\}", wikitext, re.DOTALL)
    if not match:
        return ""
    grid_text = match.group("content")
    slots = re.findall(r"\|([ABC][123])=([^|\n]+)", grid_text)
    output_match = re.search(r"\|Output=([^|\n]+)", grid_text)
    lines = [f"{slot}: {item.strip()}" for slot, item in slots if item.strip()]
    if output_match:
        lines.append(f"Output: {output_match.group(1).strip()}")
    return "\n".join(lines)


def build_dataset(items: List[str]) -> Dataset:
    """Create a ``Dataset`` of crafting questions and answers from the wiki."""
    rows: List[Dict[str, Any]] = []
    for item in items:
        wt = fetch_wikitext(item)
        recipe = parse_crafting_grid(wt)
        if recipe:
            rows.append({
                "question": f"How do you craft {item}?",
                "answer": recipe,
            })
    return Dataset.from_list(rows)


# ----------------------------------------------------------------------------
# Reward and environment setup
# ----------------------------------------------------------------------------

parser = vf.Parser()


def recipe_reward(completion: vf.Messages | str, answer: str, **_: Any) -> float:
    """Simple string matching reward for correct recipe mention."""
    text = parser.parse_answer(completion) or ""
    return 1.0 if answer.lower() in text.lower() else 0.0


rubric = vf.Rubric(
    funcs=[recipe_reward, parser.get_format_reward_func()],
    weights=[1.0, 0.2],
)

system_prompt = (
    "You are a Minecraft crafting assistant."
    " Respond step by step inside <think>...</think> tags and give the final"
    " recipe in <answer>...</answer> tags."
)

# Example dataset from a few items. Extend this list as needed.
ITEMS = [
    "Wooden Pickaxe",
    "Stone Pickaxe",
    "Cake",
]

dataset = build_dataset(ITEMS)

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    message_type="chat",
)

# ----------------------------------------------------------------------------
# GRPO training
# ----------------------------------------------------------------------------

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
training_args = vf.grpo_defaults(run_name="minecraft-crafting")

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)

trainer.train()
