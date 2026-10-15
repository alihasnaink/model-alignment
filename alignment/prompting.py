"""The single prompt format shared by SFT, DPO, and generation.

Qwen2.5-0.5B is a *base* model, so a plain-text instruction/response template is
used rather than a chat template. Keeping this in one module guarantees that
training and evaluation tokenize prompts identically.
"""

from __future__ import annotations

PROMPT_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"

# Markers that indicate the model has started a new turn on its own. Generations
# are cut there; the same rule is applied to every model so comparisons stay fair.
STOP_MARKERS = ("\n### Instruction:", "\n### Response:", "\n### Input:")


def format_prompt(prompt: str, template: str = PROMPT_TEMPLATE) -> str:
    return template.format(prompt=str(prompt).strip())


def clean_generation(text: str) -> str:
    cut = len(text)
    for marker in STOP_MARKERS:
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].strip()
