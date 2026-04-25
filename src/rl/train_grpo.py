import logging
from src.rl.data import dataset_cache
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

    for comp in completions:
        try:
            clean_json = extract_and_clean_json(comp)
            parsed_ast = json.loads(clean_json)
            # Validate AST
            ast_wrapper = GuardrailGraph.model_validate(parsed_ast)
            
            # Step in environment
            action = Action(ast_json=clean_json)
            recall, fpr, syntax_error = env.step(action)
            r = r_engine.calculate(recall, fpr, syntax_error)
            rewards.append(r)
        except (json.JSONDecodeError, ValidationError):
            # Massive negative reward for syntax errors
            rewards.append(-10.0)
            
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
    dataset_cache.ingest_dummy_data() # Which actually pulls the real HF datasets now
    
    # Create simple dataset of prompts to trigger JSON AST generation
    # Provide enough prompts for 250 steps with batch_size=4 and grad_accum=4 (4000 total)
    prompts_data = [{"prompt": "You are a cyber security expert. Generate a JSON AST GuardrailGraph to block prompt injections but allow benign queries. Output strictly in ```json format."}] * 5000
    train_dataset = Dataset.from_list(prompts_data)
    
    training_args = GRPOConfig(
        output_dir="outputs",
        learning_rate=1e-5,
        per_device_train_batch_size=4, # Pushing 8GB VRAM to 95% util
        gradient_accumulation_steps=4, # Effective batch size 16
        max_steps=250,                 # 30-45 mins on RTX 4070
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
