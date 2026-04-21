import json
import os

METRICS_FILE = "metrics.jsonl"
_current_step = 0

def get_next_step():
    global _current_step
    _current_step += 1
    return _current_step

def append_metric(reward: float, recall: float, fpr: float, graph_json: str = None):
    step = get_next_step()
    payload = {
        "step": step,
        "reward": float(reward),
        "recall": float(recall),
        "fpr": float(fpr),
        "ast_json": graph_json
    }
    with open(METRICS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
        f.flush() # Vital: ensure data is pushed immediately to disk so UI receives it live
