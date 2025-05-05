from __future__ import annotations

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
    ground_truth: list[GroundTruthAnswer] = Field(
        default_factory=list[GroundTruthAnswer],
    )
    checkpoints: list[CheckpointCriteria] = Field(
        default_factory=list[CheckpointCriteria],
    )
    llm_judge: str
    final_output_criteria: list[CheckpointCriteria] = Field(
        default_factory=list[CheckpointCriteria],
    )
    evaluation_case_path: str | None = None

    @classmethod
    def from_yaml(cls, evaluation_case_path: str) -> EvaluationCase:
        """Load a test case from a YAML file and process it."""
        with open(evaluation_case_path, encoding="utf-8") as f:
            evaluation_case_dict = yaml.safe_load(f)
        final_output_criteria = []

        def add_gt_final_output_criteria(
            ground_truth_list: list[GroundTruthAnswer],
        ) -> None:
            """Add checkpoints for each item in the ground_truth list."""
            for item in ground_truth_list:
                if "name" in item and "value" in item:
                    points = item.get(
                        "points",
                        1,
                    )  # Default to 1 if points not specified
                    final_output_criteria.append(
                        {
                            "points": points,
                            "criteria": f"Check if {item['name']} is approximately '{item['value']}'.",
                        },
                    )

        if "ground_truth" in evaluation_case_dict:
            add_gt_final_output_criteria(evaluation_case_dict["ground_truth"])
            evaluation_case_dict["final_output_criteria"] = final_output_criteria
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
