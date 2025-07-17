"""Comprehensive Minecraft crafting example using verifiers.

This script is designed to be read top to bottom. It covers data collection from
Minecraft Wiki, environment construction, and GRPO training. No modifications to
the verifiers library are required; everything here uses the existing APIs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import requests
from datasets import Dataset

import verifiers as vf

# ---------------------------------------------------------------------------
# 1. Data collection from the Minecraft Wiki
# ---------------------------------------------------------------------------

API_URL = "https://minecraft.wiki/api.php"


def fetch_wikitext(page: str) -> str:
    """Return raw wikitext for a wiki page."""
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
    """Extract a plain text recipe from a Grid/Crafting Table template."""
    match = re.search(
        r"\{\{[Gg]rid/Crafting Table(?P<content>.*?)\}\}", wikitext, re.DOTALL
    )
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
    """Build a Dataset of crafting questions and answers."""
    rows: List[Dict[str, Any]] = []
    for item in items:
        wt = fetch_wikitext(item)
        recipe = parse_crafting_grid(wt)
        if recipe:
            rows.append(
                {
                    "question": f"How do you craft {item}?",
                    "answer": recipe,
                }
            )
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# 2. Parser and rubric definitions
# ---------------------------------------------------------------------------

parser = vf.Parser()


def recipe_reward(completion: vf.Messages | str, answer: str, **_: Any) -> float:
    """Reward exact mention of the expected recipe text."""
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

# Example dataset items. Add any wiki page title here.
ITEMS = ["Wooden Pickaxe", "Stone Pickaxe", "Cake"]


# ---------------------------------------------------------------------------
# 3. Environment setup
# ---------------------------------------------------------------------------

dataset = build_dataset(ITEMS)

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    message_type="chat",
)

# ---------------------------------------------------------------------------
# 4. GRPO training
# ---------------------------------------------------------------------------

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
training_args = vf.grpo_defaults(run_name="minecraft-walkthrough")

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)

if __name__ == "__main__":
    trainer.train()
