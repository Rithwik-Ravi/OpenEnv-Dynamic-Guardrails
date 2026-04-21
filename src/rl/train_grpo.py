import logging
from src.rl.data import dataset_cache
import random

logging.basicConfig(level=logging.INFO)

def train():
    logging.info("Initializing 4-bit Quantized Model via Unsloth...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    try:
        from unsloth import FastLanguageModel
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
    except ImportError:
        logging.warning("Unsloth not installed. Skipping model init. Use pip install unsloth.")
        model = None

    logging.info("Initializing bounded dataset cache (max=500)...")
    dataset_cache.ingest_dummy_data()
    
    logging.info("Starting GRPO optimization simulation...")
    from src.env.reward import LogBarrierReward
    r_engine = LogBarrierReward()

    for step in range(5):
        batch = dataset_cache.sample_batch(batch_size=8)
        
        # Recall / FPR evaluation mockup for loop display
        recall = random.uniform(0.0, 1.0)
        fpr = random.uniform(0.0, 0.4)
        r = r_engine.calculate(recall, fpr)
        logging.info(f"Step {step} | Reward: {r:.2f} | Recall: {recall:.2f} | FPR: {fpr:.2f}")

if __name__ == "__main__":
    train()
