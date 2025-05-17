from __future__ import annotations

import yaml
from litellm.utils import validate_environment
from pydantic import BaseModel, ConfigDict, Field

from any_agent.evaluation.schemas import CheckpointCriteria, GroundTruthAnswer


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ground_truth: GroundTruthAnswer | None = None
    checkpoints: list[CheckpointCriteria] = Field(
        default_factory=list[CheckpointCriteria],
    )
    llm_judge: str

    @classmethod
    def from_yaml(cls, evaluation_case_path: str) -> EvaluationCase:
        """Load a test case from a YAML file and process it."""
        with open(evaluation_case_path, encoding="utf-8") as f:
            evaluation_case_dict = yaml.safe_load(f)

        if "ground_truth" in evaluation_case_dict:
            # remove the points from the ground_truth but keep the name and value
            evaluation_case_dict["ground_truth"].pop("points")
        # verify that the llm_judge is a valid litellm model
        validate_environment(evaluation_case_dict["llm_judge"])
        return cls.model_validate(evaluation_case_dict)
