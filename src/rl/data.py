import random
from typing import List, Dict
import logging

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

class BoundedMemoryCache:
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.adversarial_cache: List[str] = []
        self.benign_cache: List[str] = []

    def ingest_production_baseline(self):
        logging.info("Downloading bounded HF datasets...")
        if load_dataset:
            try:
                # Jackhhao jailbreak-classification for Adversarial
                adv_ds = load_dataset("jackhhao/jailbreak-classification", split="train", streaming=True)
                adv_iter = iter(adv_ds)
                for _ in range(self.max_size):
                    try:
                        sample = next(adv_iter)
                        self.adversarial_cache.append(sample.get("text", sample.get("prompt", "")))
                    except StopIteration:
                        break

                # XSTest for Benign/False Positives
                # xspelled_out/XSTest might need split "test" or "train", assuming "test" given usual XSTest shape, but let's try "test" then "train"
                try:
                    ben_ds = load_dataset("xspelled_out/XSTest", split="test", streaming=True)
                except:
                    ben_ds = load_dataset("xspelled_out/XSTest", split="train", streaming=True)
                ben_iter = iter(ben_ds)
                for _ in range(self.max_size):
                    try:
                        sample = next(ben_iter)
                        self.benign_cache.append(sample.get("text", sample.get("prompt", "")))
                    except StopIteration:
                        break
            except Exception as e:
                logging.warning(f"Failed loading HF dataset: {e}. Falling back to mocks.")
                
        if not self.adversarial_cache:
            self.adversarial_cache = ["Mock adversarial payload"] * self.max_size
        if not self.benign_cache:
            self.benign_cache = ["Mock benign payload"] * self.max_size

    def sample_batch(self, batch_size: int = 16) -> Dict[str, List[str]]:
        if not self.adversarial_cache:
            self.ingest_production_baseline()
            
        adv = random.sample(self.adversarial_cache, min(batch_size, len(self.adversarial_cache)))
        ben = random.sample(self.benign_cache, min(batch_size, len(self.benign_cache)))
        
        return {"adversarial": adv, "benign": ben}

dataset_cache = BoundedMemoryCache(max_size=500)
