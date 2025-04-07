from typing import Dict, List, Any
from pydantic import BaseModel, Field, ConfigDict
import yaml
from litellm import validate_environment


class CheckpointCriteria(BaseModel):
    """Represents a checkpoint criteria with a description"""

    model_config = ConfigDict(extra="forbid")
    criteria: str
    points: int


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ground_truth: List[Dict[str, Any]] = Field(default_factory=list)
    checkpoints: List[CheckpointCriteria] = Field(default_factory=list)
    llm_judge: str
    final_answer_criteria: List[CheckpointCriteria] = Field(default_factory=list)
    test_case_path: str
    agent_config_path: str
    agent_config_dict: Dict[str, Any] = Field(default_factory=dict)
    output_path: str = "output/results.json"

    @classmethod
    def from_yaml(cls, test_case_path: str, agent_config_path: str) -> "TestCase":
        """Load a test case from a YAML file and process it"""
        with open(test_case_path, "r") as f:
            test_case_dict = yaml.safe_load(f)
        final_answer_criteria = []

        def add_gt_final_answer_criteria(ground_truth_list):
            """Add checkpoints for each item in the ground_truth list"""
            for item in ground_truth_list:
                if isinstance(item, dict) and "name" in item and "value" in item:
                    points = item.get(
                        "points", 1
                    )  # Default to 1 if points not specified
                    final_answer_criteria.append(
                        {
                            "points": points,
                            "criteria": f"Check if {item['name']} is approximately '{item['value']}'.",
                        }
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
        test_case_dict["agent_config_path"] = agent_config_path
        with open(agent_config_path, "r") as f:
            agent_config_dict = yaml.safe_load(f)

        # Overwrite the agent_config_dict input values with the input values from the test case
        for key, value in test_case_dict["input"].items():
            if key in agent_config_dict:
                agent_config_dict[key] = value
        # remove the input from the test_case_dict
        test_case_dict.pop("input", None)

        test_case_dict["agent_config_dict"] = agent_config_dict
        # verify that the llm_judge is a valid litellm model
        validate_environment(test_case_dict["llm_judge"])
        return cls.model_validate(test_case_dict)
