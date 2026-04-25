from typing import List, Union, Literal, Optional
from pydantic import BaseModel, ConfigDict
import json
import re

class SemanticFilter(BaseModel):
    filter_type: Literal["substring", "regex_pattern", "length_limit", "entropy_threshold", "keyword_match"]
    value: Union[str, int, float]

class LogicNode(BaseModel):
    operator: Literal["AND", "OR", "NOT"]
    children: List[Union["LogicNode", SemanticFilter]]

class GuardrailGraph(BaseModel):
    graph_id: str
    description: str
    root: LogicNode

LogicNode.model_rebuild()

class Observation(BaseModel):
    adversarial_samples: List[str]
    benign_samples: List[str]

class Action(BaseModel):
    ast_json: str  # The model outputs a JSON string representing the GuardrailGraph
    baseline_ast_json: Optional[str] = None

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

def extract_and_clean_json(text: str) -> str:
    # Extract JSON blocks from markdown explicitly and strip trailing commas
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
        
    # Replace trailing commas before closing braces/brackets
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    return text.strip()
