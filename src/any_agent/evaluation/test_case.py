from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003

import yaml
from litellm.utils import validate_environment
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


class CheckpointCriteria(BaseModel):
    """Represents a checkpoint criteria with a description"""

    model_config = ConfigDict(extra="forbid")
    criteria: str
    points: int


class GroundTruthAnswer(TypedDict):
    name: str
    value: float
    points: float


class TestCase(BaseModel):
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
    test_case_path: str
    output_path: str = "output/results.json"

    @classmethod
    def from_yaml(cls, test_case_path: str) -> TestCase:
        """Load a test case from a YAML file and process it"""
        with open(test_case_path) as f:
            test_case_dict = yaml.safe_load(f)
        final_answer_criteria = []

        def add_gt_final_answer_criteria(
            ground_truth_list: Sequence[GroundTruthAnswer],
        ) -> None:
            """Add checkpoints for each item in the ground_truth list"""
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

        if "ground_truth" in test_case_dict:
            add_gt_final_answer_criteria(test_case_dict["ground_truth"])
            test_case_dict["final_answer_criteria"] = final_answer_criteria
            # remove the points from the ground_truth list but keep the name and value
            test_case_dict["ground_truth"] = [
                item
                for item in test_case_dict["ground_truth"]
                if isinstance(item, dict)
            ]

        test_case_dict["test_case_path"] = test_case_path
        # verify that the llm_judge is a valid litellm model
        validate_environment(test_case_dict["llm_judge"])
        return cls.model_validate(test_case_dict)
