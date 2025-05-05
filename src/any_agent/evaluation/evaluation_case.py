from __future__ import annotations

# This can't go into a TYPE_CHECKING block because it's used at runtime by Pydantic to build the fields
from collections.abc import Sequence  # noqa: TC003

import yaml
from litellm.utils import validate_environment
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


class CheckpointCriteria(BaseModel):
    """Represents a checkpoint criteria with a description."""

    model_config = ConfigDict(extra="forbid")
    criteria: str
    points: int


class GroundTruthAnswer(TypedDict):
    name: str
    value: float
    points: float


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ground_truth: Sequence[GroundTruthAnswer] = Field(
        default_factory=list[GroundTruthAnswer],
    )
    checkpoints: Sequence[CheckpointCriteria] = Field(
        default_factory=list[CheckpointCriteria],
    )
    llm_judge: str
    final_answer_criteria: Sequence[CheckpointCriteria] = Field(
        default_factory=list[CheckpointCriteria],
    )
    evaluation_case_path: str | None = None

    @classmethod
    def from_yaml(cls, evaluation_case_path: str) -> EvaluationCase:
        """Load a test case from a YAML file and process it."""
        with open(evaluation_case_path, encoding="utf-8") as f:
            evaluation_case_dict = yaml.safe_load(f)
        final_answer_criteria = []

        def add_gt_final_answer_criteria(
            ground_truth_list: Sequence[GroundTruthAnswer],
        ) -> None:
            """Add checkpoints for each item in the ground_truth list."""
            for item in ground_truth_list:
                if "name" in item and "value" in item:
                    points = item.get(
                        "points",
                        1,
                    )  # Default to 1 if points not specified
                    final_answer_criteria.append(
                        {
                            "points": points,
                            "criteria": f"Check if {item['name']} is approximately '{item['value']}'.",
                        },
                    )

        if "ground_truth" in evaluation_case_dict:
            add_gt_final_answer_criteria(evaluation_case_dict["ground_truth"])
            evaluation_case_dict["final_answer_criteria"] = final_answer_criteria
            # remove the points from the ground_truth list but keep the name and value
            evaluation_case_dict["ground_truth"] = [
                item
                for item in evaluation_case_dict["ground_truth"]
                if isinstance(item, dict)
            ]

        evaluation_case_dict["evaluation_case_path"] = evaluation_case_path
        # verify that the llm_judge is a valid litellm model
        validate_environment(evaluation_case_dict["llm_judge"])
        return cls.model_validate(evaluation_case_dict)
