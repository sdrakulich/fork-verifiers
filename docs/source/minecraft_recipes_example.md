# Minecraft Recipes Example

This walkthrough highlights the key abstractions provided by the
`verifiers` framework while building a small reinforcement learning
environment for answering Minecraft crafting questions.

## Overview of Abstractions

- **Parser** – Extracts structured fields from model outputs. The
  `XMLParser` is used to split `<think>` reasoning from the final
  `<answer>`.
- **Rubric** – A weighted collection of reward functions. Here we check
  both correctness of the extracted answer and compliance with the
  required format.
- **Environment** – Manages prompts, model interactions and scoring.
  `SingleTurnEnv` is used for one question/answer per example.
- **Trainer** – `GRPOTrainer` performs reinforcement learning with
  Group‑Relative Policy Optimization.

## Building the Dataset

Recipes are sourced directly from the [Minecraft wiki](https://minecraft.wiki).
`build_dataset` downloads the HTML for each item via the MediaWiki API and
extracts the crafting section with BeautifulSoup. Each row contains a
`question` like "How do you craft a cake?" and the authoritative
`answer` text from the wiki.

````python
from datasets import Dataset
from typing import List
import requests
from bs4 import BeautifulSoup

WIKI_API = "https://minecraft.wiki/api.php"

def fetch_page_html(title: str) -> str:
    params = {"action": "parse", "page": title, "prop": "text", "format": "json"}
    res = requests.get(WIKI_API, params=params, timeout=30)
    res.raise_for_status()
    return res.json()["parse"]["text"]["*"]

def parse_crafting_section(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    header = soup.find(id="Crafting")
    for sibling in header.next_siblings:
        if getattr(sibling, "name", None) == "h2":
            break
        text = getattr(sibling, "get_text", lambda **_: "")()
        yield text.strip()

def build_dataset(items: List[str]) -> Dataset:
    rows = []
    for item in items:
        html = fetch_page_html(item)
        recipe = " ".join(parse_crafting_section(html))
        rows.append({"question": f"How do you craft {item}?", "answer": recipe})
    return Dataset.from_list(rows)
````

## Setting Up the Environment

````python
import verifiers as vf

parser = vf.XMLParser(["think", "answer"])

def correctness(completion, answer, **_):
    response = parser.parse_answer(completion) or ""
    return 1.0 if response.strip().lower() == answer.strip().lower() else 0.0

rubric = vf.Rubric(
    funcs=[correctness, parser.get_format_reward_func()],
    weights=[1.0, 0.2],
)

system_prompt = (
    "You are a Minecraft crafting expert."
    " Respond using <think> and <answer> tags."
)

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
)
````

## Training with GRPO

````python
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
args = vf.grpo_defaults(run_name="minecraft_recipes")

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)
trainer.train()
````

Running this script trains the model to output accurate crafting recipes
in the required format directly from wiki-sourced data.
