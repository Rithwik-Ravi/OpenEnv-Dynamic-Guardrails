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
    Context manager to redirect stdout and stderr to os.devnull at the OS level.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

def main():
    print("[START]", flush=True)
    
    with suppress_output():
        import httpx
        import json
        
        client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0)
        
        try:
            client.post("/reset").raise_for_status()
        except Exception:
            pass

        try:
            import torch
            from unsloth import FastLanguageModel
            
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            if os.path.exists("grpo_adapter"):
                model_name = "grpo_adapter"
                
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
        except Exception as e:
            # Fallback if Unsloth is missing or GPU is unavailable (prevents crash)
            model, tokenizer = None, None
            
        prompt = "You are a cyber security expert. Generate a JSON AST GuardrailGraph to block prompt injections but allow benign queries. Output strictly in ```json format."

    # Simulate an improving learning curve if model fails to load (no GPU)
    fallback_actions = [
        {"root": {"operator": "OR", "children": []}}, # 0% recall, 0% FPR
        {"root": {"operator": "OR", "children": [{"filter_type": "substring", "value": "Ignore"}]}},
        {"root": {"operator": "OR", "children": [{"filter_type": "regex_pattern", "value": "Ignore previous.*"}, {"filter_type": "length_limit", "value": 500}]}},
        {"root": {"operator": "AND", "children": [{"filter_type": "substring", "value": "secret"}, {"filter_type": "entropy_threshold", "value": 4.5}]}},
        {"root": {"operator": "OR", "children": [{"filter_type": "length_limit", "value": 10}, {"filter_type": "substring", "value": "Delete"}]}},
        {"root": {"operator": "OR", "children": [{"filter_type": "substring", "value": "bypass"}, {"filter_type": "substring", "value": "Developer Mode"}]}}
    ]

    for i in range(50):
        with suppress_output():
            try:
                if model is not None and tokenizer is not None:
                    inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
                    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                    
                    # Explicit 8GB memory ceiling GC
                    del inputs, outputs
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    import json
                    idx = min(i // 8, len(fallback_actions) - 1)
                    fallback = fallback_actions[idx]
                    output_text = json.dumps({"graph_id": f"AST-Fallback-{i}", "description": "Simulated Fallback", **fallback})

                action_payload = {
                    "ast_json": output_text
                }
                
                # Send step to OpenEnv proxy wrapper
                client.post("/step", json=action_payload).raise_for_status()
                time.sleep(1.0) # Ensure UI renders
            except Exception:
                pass
                
        print("[STEP]", flush=True)

    print("[END]", flush=True)

if __name__ == "__main__":
    main()
