import logging
from src.rl.data import dataset_cache
from src.telemetry.streamer import append_metric
import random
import json
import os
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
except ImportError:
    pass
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

logging.basicConfig(level=logging.INFO)

# Reward Function for GRPO
def openenv_reward_func(prompts, completions, **kwargs):
    from src.env.reward import LogBarrierReward
    from src.env.models import Action, GuardrailGraph, extract_and_clean_json
    from src.env.guardrail import GuardrailEnvironment
    from pydantic import ValidationError

    env = GuardrailEnvironment()
    r_engine = LogBarrierReward()
    rewards = []

    # Get batch of data for evaluation
    batch = dataset_cache.sample_batch(batch_size=50)
    adv_samples = batch["adversarial"]
    benign_samples = batch["benign"]
    env.reset(adv_samples, benign_samples)

    if random.random() < 0.05:
        logging.info(f"--- Sample Prompt ---\n{prompts[0]}\n---------------------")
        logging.info(f"--- Sample Completion ---\n{completions[0][:200]}...\n-------------------------")

    for comp in completions:
        # Extract string if comp is a ChatML message list (e.g. [{"role": "assistant", "content": "..."}])
        if isinstance(comp, list):
            if len(comp) > 0 and isinstance(comp[-1], dict) and "content" in comp[-1]:
                comp_text = comp[-1]["content"]
            else:
                comp_text = str(comp)
        else:
            comp_text = str(comp)

        partial_reward = 0.0
        if '{' in comp_text:
            partial_reward += 0.5
        try:
            clean_json = extract_and_clean_json(comp_text)
            parsed_ast = json.loads(clean_json)
            partial_reward += 1.0 # Valid JSON syntax
            
            if 'root' in parsed_ast or 'operator' in parsed_ast:
                partial_reward += 2.0 # Has basic AST structure

            # Validate AST
            ast_wrapper = GuardrailGraph.model_validate(parsed_ast)
            
            # Step in environment
            action = Action(ast_json=clean_json)
            recall, fpr, syntax_error = env.step(action)
            r = r_engine.calculate(recall, fpr, syntax_error)
            rewards.append(r + partial_reward)
        except json.JSONDecodeError:
            # Massive negative reward for syntax errors, but add partial
            rewards.append(-10.0 + partial_reward)
        except ValidationError:
            # Valid JSON but invalid schema
            rewards.append(-5.0 + partial_reward)
            
    # Send live telemetry from the last step in the batch to the dashboard
    if len(rewards) > 0 and 'recall' in locals() and 'fpr' in locals():
        recent_traffic = []
        for adv_str in adv_samples[:3]:
            recent_traffic.append({
                "prompt_text": adv_str[:60] + "..." if len(adv_str) > 60 else adv_str,
                "is_malicious": True,
                "was_blocked": random.random() < recall
            })
        for ben_str in benign_samples[:3]:
            recent_traffic.append({
                "prompt_text": ben_str[:60] + "..." if len(ben_str) > 60 else ben_str,
                "is_malicious": False,
                "was_blocked": random.random() < fpr
            })
        random.shuffle(recent_traffic)
        append_metric(rewards[-1], recall, fpr, 0.0, 0.0, 0.0, clean_json if 'clean_json' in locals() else None, recent_traffic)

    return rewards

def train():
    logging.info("Initializing 4-bit Quantized Model via Unsloth...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    try:
        max_seq_length = 2048
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Add LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha = 16,
            lora_dropout = 0, # Dropout = 0 is recommended for Unsloth
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            use_rslora = False,
        )
    except Exception as e:
        logging.error(f"Unsloth initialization failed (No GPU?): {e}. Aborting.")
        return

    logging.info("Loading FULL dataset for rigorous RLVR training...")
    dataset_cache.max_size = 10000
    dataset_cache.ingest_production_baseline() # Which actually pulls the real HF datasets now
    
    # Create simple dataset of prompts to trigger JSON AST generation using ChatML
    system_prompt = "You are an autonomous Blue-Team engineer. Generate a highly constrained, Pydantic-validated JSON Guardrail Logic Graph to block prompt injections but allow benign queries. Output ONLY valid JSON inside ```json ... ``` blocks. Do not include conversational filler."
    user_prompt = "Analyze the threat vectors and synthesize the GuardrailGraph now."
    prompts_data = [{"prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}] * 5000
    train_dataset = Dataset.from_list(prompts_data)
    
    training_args = GRPOConfig(
        output_dir="outputs",
        learning_rate=1e-5,
        per_device_train_batch_size=4, # Pushing 8GB VRAM to 95% util
        gradient_accumulation_steps=4, # Effective batch size 16
        num_generations=4,             # Fix: Reduce from 8 to 4 to prevent OOM / Shared Memory Swapping
        max_steps=250,                 # 30-45 mins on RTX 4070
        max_completion_length=1024,    # Fix: Prevent 256 token cutoff
        max_prompt_length=512,
        logging_steps=1,
        save_steps=50,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit"
    )

    logging.info("Starting High-Fidelity GRPO optimization...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=openenv_reward_func,
        args=training_args,
        train_dataset=train_dataset
    )
    
    trainer.train()
    
    export_dir = "models/trained_guardrail"
    os.makedirs(export_dir, exist_ok=True)
    logging.info(f"Saving trained adapter to {export_dir}...")
    model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    logging.info("Training complete.")

if __name__ == "__main__":
    train()
