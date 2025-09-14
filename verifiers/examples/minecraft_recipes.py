import os
import requests
from bs4 import BeautifulSoup
from datasets import Dataset
from typing import List

import verifiers as vf

"""Example: Train a model to answer Minecraft crafting questions.

This script demonstrates the main abstractions of the ``verifiers`` framework:
    * ``Parser`` -- extracts fields from model output.
    * ``Rubric`` -- scores outputs with reward functions.
    * ``SingleTurnEnv`` -- orchestrates prompts, completions and rewards.
    * ``GRPOTrainer`` -- reinforcement learning loop.

Dataset rows are scraped from the official Minecraft wiki using its MediaWiki API.
Network access is required to run ``build_dataset``. Each question asks how to
craft an item, and the answer contains the recipe text extracted from the wiki.
The model must respond using ``<think>`` and ``<answer>`` tags.
"""

WIKI_API = "https://minecraft.wiki/api.php"


def fetch_page_html(title: str) -> str:
    """Download HTML for a wiki page."""
    params = {"action": "parse", "page": title, "prop": "text", "format": "json"}
    response = requests.get(WIKI_API, params=params, timeout=30)
    response.raise_for_status()
    return response.json()["parse"]["text"]["*"]


def parse_crafting_section(html: str) -> str:
    """Extract the crafting section text from a page's HTML."""
    soup = BeautifulSoup(html, "html.parser")
    crafting_header = soup.find(id="Crafting")
    if not crafting_header:
        raise ValueError("Crafting section not found")
    section_text = []
    for sibling in crafting_header.next_siblings:
        if getattr(sibling, "name", None) == "h2":
            break
        section_text.append(getattr(sibling, "get_text", lambda **_: "")())
    return " ".join(t.strip() for t in section_text if t.strip())


def build_dataset(items: List[str]) -> Dataset:
    """Create a HuggingFace ``Dataset`` from wiki recipes."""
    rows = []
    for item in items:
        html = fetch_page_html(item)
        recipe = parse_crafting_section(html)
        question = f"How do you craft {item.replace('_', ' ')}?"
        rows.append({"question": question, "answer": recipe})
    return Dataset.from_list(rows)


# Example list of pages. Extend as needed.
ITEMS = ["Crafting_Table", "Furnace", "Cake"]


def main() -> None:
    dataset = build_dataset(ITEMS)

    parser = vf.XMLParser(["think", "answer"])

    def correctness(completion, answer, **_) -> float:
        response = parser.parse_answer(completion) or ""
        return 1.0 if response.strip().lower() == answer.strip().lower() else 0.0

    rubric = vf.Rubric(
        funcs=[correctness, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
    )

    system_prompt = f"""You are a Minecraft crafting expert. Answer user questions in the following format:
{parser.get_format_str()}"""

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    model_name = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    args = vf.grpo_defaults(run_name="minecraft_recipes")

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
