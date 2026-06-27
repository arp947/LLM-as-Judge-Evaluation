from pydantic import BaseModel, Field
from typing import Literal, List

class CriterionVerdict(BaseModel):
    criterion_name: str = Field(description="Name of the evaluated aspect: e.g., Correctness, Faithfulness, Completeness")
    grounding_citations: List[str] = Field(description="Verbatim text quotes extracted from the models to ground the evaluation.")
    winner: Literal["A", "B", "TIE"] = Field(description="The preferred model output for this explicit criterion.")
    rationale: str = Field(description="Detailed objective reasoning showing why one outperformed the other based on the rubric.")

class StructuredJudgeVerdict(BaseModel):
    per_criterion_eval: List[CriterionVerdict] = Field(description="Breakdown per item of the evaluation rubric.")
    overall_winner: Literal["A", "B", "TIE"] = Field(description="Final balanced structural preference choice.")
    global_rationale: str = Field(description="Consolidated comprehensive logic explaining the ultimate verdict.")
