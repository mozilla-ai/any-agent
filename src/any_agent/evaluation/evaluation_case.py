from __future__ import annotations

from uuid import uuid4

import yaml
from litellm.utils import validate_environment
from pydantic import BaseModel, ConfigDict, Field, field_validator

from any_agent.evaluation.schemas import CheckpointCriteria, GroundTruthAnswer


class EvaluationCase(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
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

    @field_validator("checkpoints", mode="after")
    @classmethod
    def validate_checkpoints(
        cls, v: list[CheckpointCriteria]
    ) -> list[CheckpointCriteria]:
        """Validate that the checkpoints are unique by id."""
        if len(v) != len({checkpoint.id for checkpoint in v}):
            msg = "Checkpoints must be unique by id."
            raise ValueError(msg)
        return v
