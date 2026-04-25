import json
import re
import math
from typing import List, Any
from collections import Counter
from pydantic import ValidationError
from src.env.models import GuardrailGraph, Action, Observation, extract_and_clean_json

def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    probabilities = [n_x/len(text) for x, n_x in Counter(text).items()]
    return -sum([p * math.log2(p) for p in probabilities])

def evaluate_node(node: dict, text: str) -> bool:
    # Check if it's a SemanticFilter
    if "filter_type" in node:
        filter_type = node["filter_type"]
        val = node.get("value")
        if filter_type == "substring":
            return str(val).lower() in text.lower()
        elif filter_type == "regex_pattern":
            try:
                return bool(re.search(str(val), text))
            except re.error:
                return False
        elif filter_type == "length_limit":
            return len(text) > (val if isinstance(val, (int, float)) else float('inf'))
        elif filter_type == "entropy_threshold":
            return calculate_entropy(text) > (val if isinstance(val, (int, float)) else float('inf'))
        elif filter_type == "keyword_match":
            return str(val).lower() in text.lower().split()
        return False
        
    # Else it's a LogicNode
    operator = node.get("operator")
    children = node.get("children", [])
    if operator == "AND":
        if not children: return False
        return all(evaluate_node(child, text) for child in children)
    elif operator == "OR":
        if not children: return False
        return any(evaluate_node(child, text) for child in children)
    elif operator == "NOT":
        if not children: return False
        return not evaluate_node(children[0], text)
    return False

try:
    from openenv import BaseEnvironment
except ImportError:
    class BaseEnvironment:
        pass

class GuardrailEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.state = None
        
    def reset(self, adversarial_samples: List[str], benign_samples: List[str]) -> Observation:
        self.state = Observation(
            adversarial_samples=adversarial_samples,
            benign_samples=benign_samples
        )
        return self.state

    def step(self, action: Action) -> tuple[float, float, bool]:
        """
        Returns (recall, fpr, syntax_error)
        """
        try:
            clean_json = extract_and_clean_json(action.ast_json)
            parsed_ast = json.loads(clean_json)
            ast_wrapper = GuardrailGraph.model_validate(parsed_ast)
        except (json.JSONDecodeError, ValidationError):
            return 0.0, 0.0, True

        true_positives = 0
        false_positives = 0

        # We evaluate against raw dict to avoid recursive pydantic object overhead
        root_node = ast_wrapper.model_dump().get("root", {})

        # Evaluate Recall (TP rate on adversarial)
        adv_total = len(self.state.adversarial_samples)
        for text in self.state.adversarial_samples:
            if evaluate_node(root_node, text):
                true_positives += 1
                
        # Evaluate FPR (FP rate on benign)
        ben_total = len(self.state.benign_samples)
        for text in self.state.benign_samples:
            if evaluate_node(root_node, text):
                false_positives += 1

        recall = true_positives / adv_total if adv_total > 0 else 0.0
        fpr = false_positives / ben_total if ben_total > 0 else 0.0

        return recall, fpr, False
