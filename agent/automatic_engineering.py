import json
from dataclasses import dataclass
from typing import Dict, List

from openai import OpenAI  # or AsyncOpenAI if you prefer async

# --- Config ------------------------------------------------------------------

GEN_MODEL = "your-generation-model"     # the model you use to write reports
OPT_MODEL = "your-optimizer-model"      # can be same or stronger model

client = OpenAI(
    base_url="http://localhost:8000/v1",  # OpenAI-compatible endpoint
    api_key="token-abc123",               # dummy for vLLM, must be non-empty
)

CURRENT_PROMPT = """<your current prompt template here, with {fields} placeholders>"""

TASK_DESCRIPTION = "Generate internal audit closure reports from structured fields in French, in a formal bank style."

@dataclass
class EvalExample:
    fields: Dict[str, str]
    expected_report: str

EVAL_EXAMPLES: List[EvalExample] = [
    # Fill with a few labeled examples
    EvalExample(
        fields={
            "assignment_context": "...",
            "recommendation_summary": "...",
            "actions_taken": "...",
            "evidence_description": "...",
            "closure_decision": "...",
        },
        expected_report="(gold report text 1)",
    ),
    # ...
]

# --- Helpers -----------------------------------------------------------------


def run_generation(prompt_template: str, example: EvalExample) -> str:
    user_prompt = prompt_template.format(**example.fields)

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.0,
        max_tokens=800,
    )
    return resp.choices[0].message.content


def build_examples_block(prompt_template: str) -> str:
    blocks = []
    for i, ex in enumerate(EVAL_EXAMPLES, start=1):
        gen = run_generation(prompt_template, ex)
        block = f"""Example {i}
input_fields:
{json.dumps(ex.fields, ensure_ascii=False, indent=2)}

expected_report:
{ex.expected_report}

generated_report:
{gen}
---"""
        blocks.append(block)
    return "\n\n".join(blocks)


def build_optimizer_prompt(current_prompt: str) -> str:
    examples_block = build_examples_block(current_prompt)

    meta_prompt = f"""
You are an expert prompt engineer.

Your goal: improve a prompt template so that a target LLM generates
internal audit reports that closely match the expected style, structure,
and content.

1) Task description:
---
{TASK_DESCRIPTION}
---

2) Current prompt template (P0):
---
{current_prompt}
---

3) Evaluation examples with model outputs:
---
{examples_block}
---

Analyze P0 and the examples, then output JSON with keys:
- "improved_prompt": the new prompt template P1
- "rationale": brief explanation of changes
"""
    return meta_prompt


def refine_prompt_once(current_prompt: str) -> Dict[str, str]:
    meta_prompt = build_optimizer_prompt(current_prompt)

    resp = client.chat.completions.create(
        model=OPT_MODEL,
        messages=[{"role": "user", "content": meta_prompt}],
        temperature=0.2,
        max_tokens=1200,
    )
    raw = resp.choices[0].message.content
    # Expect strict JSON per the meta-prompt; you can add try/except here
    data = json.loads(raw)
    return data


if __name__ == "__main__":
    result = refine_prompt_once(CURRENT_PROMPT)
    print("Improved prompt:\n", result["improved_prompt"])
    print("\nRationale:\n", result["rationale"])
