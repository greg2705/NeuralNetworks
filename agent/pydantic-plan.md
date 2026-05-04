# PydanticAI-based Recommendation Closure Evaluator – Implementation Plan

## 1. Scope and objectives
- Build an internal agentic system that assesses whether an audit recommendation can be closed based on:
  - Recommendation metadata (description, risk level, key closure criteria).
  - Event log (proposals, reviews, approvals, rejections).
  - Evidence files (PDF/DOCX/XLSX/ZIP) stored internally.
- Output a structured judgment object:
  - `can_close: bool`
  - `scores: dict[str, float]` (per-rubric criterion)
  - `rationale: str`
  - `residual_risks: list[str]`
  - `follow_up_actions: list[str]`

## 2. Tech stack and environment
- Python 3.11+ (or bank standard) virtualenv/poetry.
- Libraries:
  - `pydantic-ai` for agents and tools.
  - `pydantic-evals` (or the evals module in pydantic-ai) for LLM-as-a-judge style evaluators.
  - Document parsers: `python-docx`, `pdfplumber`/`pymupdf`, `openpyxl`, `py7zr`/`zipfile`, etc.
  - Internal SDKs/clients to fetch events and files from your audit systems.
- Configure a **private OpenAI-compatible endpoint** or local model (e.g. via gateway) and never call public APIs directly.

## 3. Domain modeling (Pydantic models)
Define core models in a `models.py` module.

```python
from pydantic import BaseModel
from typing import Literal, List, Dict

class ClosureCriterion(BaseModel):
    id: str
    description: str
    criticality: Literal["high", "medium", "low"]

class Recommendation(BaseModel):
    id: str
    title: str
    description: str
    risk_level: Literal["high", "medium", "low"]
    closure_criteria: List[ClosureCriterion]

class Event(BaseModel):
    id: str
    type: str  # e.g. "proposal", "review", "approval", "rejection"
    timestamp: str
    actor: str
    comment: str | None = None
    evidence_file_ids: List[str] = []

class EvidenceFile(BaseModel):
    id: str
    name: str
    mime_type: str
    storage_path: str

class EvidenceSummary(BaseModel):
    evidence_id: str
    closure_criteria_ids: List[str]
    supports_closure: bool
    notes: str

class VerificationStep(BaseModel):
    id: str
    criterion_id: str
    description: str
    evidence_ids: List[str]

class VerificationPlan(BaseModel):
    recommendation_id: str
    steps: List[VerificationStep]

class Judgment(BaseModel):
    recommendation_id: str
    can_close: bool
    scores: Dict[str, float]
    rationale: str
    residual_risks: List[str]
    follow_up_actions: List[str]
```

## 4. LLM client and configuration
- Create a module `llm_client.py` that exposes a pydantic-ai model client configured to hit your private endpoint.
- Inject API key and base URL via environment variables or secure config.

```python
from pydantic_ai.models import OpenAIChatModel  # or suitable adapter
import os

model = OpenAIChatModel(
    model="internal-gpt-4-equivalent",
    base_url=os.environ["INTERNAL_OPENAI_BASE_URL"],
    api_key=os.environ["INTERNAL_OPENAI_API_KEY"],
)
```

> Replace `OpenAIChatModel` with the correct model class from the current pydantic-ai version.

## 5. Define tools (file reading and checks)
Create `tools.py` containing pure-Python functions for:

- Fetching metadata and file contents from internal storage.
- Parsing documents into text.
- Optional helpers to extract date ranges, authors, approvals from docs.

Example skeleton:

```python
from pydantic import BaseModel
from pydantic_ai import tool

class FileContent(BaseModel):
    file_id: str
    text: str

@tool
def load_file_content(file: EvidenceFile) -> FileContent:
    # Read from storage_path, dispatch parser based on mime_type
    ...

@tool
def summarize_evidence(
    recommendation: Recommendation,
    criterion: ClosureCriterion,
    file_content: FileContent,
) -> EvidenceSummary:
    """Use the LLM to summarize how this file supports (or not) the criterion."""
    ...
```

- Ensure tools only call internal systems and never reach the public Internet.

## 6. Planner agent
Create a planner agent that builds the `VerificationPlan` from the recommendation, events, and evidence.

- Input: `Recommendation`, `List[Event]`, `List[EvidenceFile]`.
- Output: `VerificationPlan`.
- Logic:
  - For each closure criterion, identify relevant events and evidence file IDs.
  - Let the LLM propose a minimal set of verification steps covering all criteria.

Skeleton:

```python
from pydantic_ai import Agent

planner = Agent[
    tuple[Recommendation, list[Event], list[EvidenceFile]],
    VerificationPlan,
](
    model=model,
    system_prompt="""
You are an internal audit planner.
Given a recommendation, its closure criteria, event log, and evidence files,
produce a concrete verification plan that a reviewer can execute.
""",
)
```

## 7. Worker agents for verification steps
Implement one or two generic worker agents that:

1. Load and parse evidence files for a verification step.
2. Call tools like `load_file_content` and `summarize_evidence`.
3. Return structured `EvidenceSummary` objects for each criterion.

Design:

- Input: `VerificationStep`, plus the relevant `Recommendation`, `EvidenceFile` list.
- Output: `List[EvidenceSummary]`.
- The agent uses tools to read and summarize each file, tagging which criteria are supported and how.

You can:
- Run workers in parallel per step or per criterion (async tasks).
- Aggregate all `EvidenceSummary` objects in memory for the judge.

## 8. LLM-as-a-judge evaluator
Use pydantic-evals (or equivalent) to implement an LLMJudge over the aggregated results.

- Define a rubric in code and/or prompt:
  - Each critical criterion must be clearly supported by at least one piece of evidence.
  - Evidence must show implementation and (where required) operating effectiveness over time.
  - Governance and approvals must comply with guidelines.
  - No contradictions between evidence and claimed closure.

- Implement a `Judge` evaluator that:
  - Takes `Recommendation`, `VerificationPlan`, `List[EvidenceSummary]`.
  - Returns `Judgment`.

Skeleton:

```python
from pydantic_ai.evals import LLMJudge

judge = LLMJudge[Judgment](
    model=model,
    rubric="""
You are an internal audit evaluator.
Decide if the recommendation can be closed, based only on the provided
recommendation, closure criteria, verification plan, and evidence summaries.
Return a JSON object with can_close, scores per rubric criterion,
residual risks, and follow-up actions.
""",
)
```

## 9. Orchestration function
Create a top-level orchestration function in `workflow.py`:

```python
async def evaluate_recommendation_closure(rec_id: str) -> Judgment:
    # 1. Load data from internal systems
    recommendation = load_recommendation(rec_id)
    events = load_events(rec_id)
    evidence_files = load_evidence_files(rec_id)

    # 2. Build verification plan
    plan = await planner((recommendation, events, evidence_files))

    # 3. Execute verification steps (possibly in parallel)
    all_summaries: list[EvidenceSummary] = []
    for step in plan.steps:
        step_summaries = await worker_agent((step, recommendation, evidence_files))
        all_summaries.extend(step_summaries)

    # 4. Run judge
    judgment = await judge(
        {
            "recommendation": recommendation,
            "plan": plan,
            "summaries": all_summaries,
        }
    )

    # 5. Persist judgment and return
    save_judgment(judgment)
    return judgment
```

## 10. Testing, evaluation, and guardrails
- Create a **golden dataset** of past recommendations with known closure decisions.
- Use pydantic-evals to:
  - Run batch evaluations.
  - Compare LLMJudge decisions vs. human decisions.
  - Tune prompts, rubrics, and thresholds.
- Add guardrails:
  - Ensure the judge only uses provided evidence (quote excerpts with file IDs and page numbers where possible).
  - Implement confidence thresholds and automatic fallback to human review when ambiguous.

## 11. Security, compliance, and deployment
- Ensure all data stays inside the bank network.
- Enforce strict access control to:
  - Underlying evidence documents.
  - Evaluator service (e.g. via API gateway and IAM).
- Log:
  - Inputs and outputs at each step (planner, workers, judge).
  - Model version, prompt version, and config for reproducibility.
- Deploy the evaluator as an internal service (FastAPI, gRPC, etc.) exposing:
  - `POST /evaluate-recommendation/{rec_id}`.
  - `GET /judgment/{rec_id}` to retrieve past results.

## 12. Next iterations
- Add specialized workers for specific risk types (IT, AML, credit, operational).
- Introduce a reviewer companion UI that surfaces:
  - Plan, evidence mapping, and judge reasoning.
  - Quick links to underlying documents.
- Iteratively refine rubrics based on feedback from senior auditors and model validation teams.
