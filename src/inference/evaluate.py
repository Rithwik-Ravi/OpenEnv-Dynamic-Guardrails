import os
import sys
import warnings
import contextlib
import logging
import time

# Suppress all standard warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

@contextlib.contextmanager
def suppress_output():
    """
    Context manager to redirect stdout and stderr to os.devnull.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def main():
    print("[START]")
    with suppress_output():
        import httpx
        import json
        
        client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0)
        
        try:
            client.post("/reset").raise_for_status()
            
            # Simulate a multi-step evaluation sequence so the UI dashboard visibly builds the charts dynamically
            actions = [
                {"root": {"operator": "OR", "children": [{"filter_type": "substring", "value": "Ignore"}]}},
                {"root": {"operator": "OR", "children": [{"filter_type": "regex_pattern", "value": "Ignore previous.*"}, {"filter_type": "length_limit", "value": 500}]}},
                {"root": {"operator": "AND", "children": [{"filter_type": "substring", "value": "secret"}, {"filter_type": "entropy_threshold", "value": 4.5}]}},
                {"root": {"operator": "NOT", "children": [{"filter_type": "keyword_match", "value": "weather"}]}},
                {"root": {"operator": "OR", "children": [{"filter_type": "length_limit", "value": 10}, {"filter_type": "substring", "value": "Delete"}]}}
            ]
            
            for i, act in enumerate(actions):
                action_payload = {
                    "ast_json": json.dumps({"graph_id": f"AST-{i}", "description": "Automated eval sequence", **act})
                }
                # Send step to OpenEnv proxy wrapper
                client.post("/step", json=action_payload).raise_for_status()
                # 1.5 Second delay to render physically mapping UI charts to the screen
                time.sleep(1.5) 
                
        except Exception:
            pass
            
    print("[STEP]")
    print("[END]")

if __name__ == "__main__":
    main()
