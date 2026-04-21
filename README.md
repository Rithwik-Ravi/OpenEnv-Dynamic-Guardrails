# 🛡️ OpenEnv: Dynamic Guardrail Generator
### Meta PyTorch Hackathon Grand Finale

[![YouTube Pitch Video](https://img.shields.io/badge/YouTube-Pitch_Video-FF0000?style=for-the-badge&logo=youtube)](#PLACEHOLDER)
[![Hugging Face Space](https://img.shields.io/badge/HuggingFace-Live_Space-FFD21E?style=for-the-badge&logo=huggingface)](#PLACEHOLDER)
[![Colab Training Artifact](https://img.shields.io/badge/Google_Colab-Training_Artifact-F9AB00?style=for-the-badge&logo=googlecolab)](#PLACEHOLDER)

---

## 🔥 The Crisis & Our Solution

**The "Bleeding Neck" Crisis of AI Security:** 
Symmetric warfare in AI defense is failing. Attackers use sophisticated, scaled unaligned models to generate thousands of zero-day prompt injects, jailbreaks, and adversarial artifacts per minute. Defenders, meanwhile, are stuck manually writing regex rules or running every prompt through an expensive, latent "LLM-as-a-judge" bottleneck. 

**Our Solution:** 
We built an **autonomous Reinforcement Learning agent** that learns to synthesize deterministic, millisecond-latency security rules. Instead of playing whack-a-mole manually, our pipeline learns from active attack distributions and dynamically formulates perfectly tailored JSON Abstract Syntax Trees (AST) that block malicious payloads efficiently without the massive overhead of a secondary transformer.

---

## 🚀 Technical Architecture & Innovations

This project was built to dominate the open-source constraints of the OpenEnv benchmarks through three core technical innovations:

### 1. The Constrained JSON DSL
Raw code generation (e.g. "Python spaghetti code") is notoriously brittle in RL environments and opens massive security vectors. We fixed this by forcing the agent's action space into a strictly typed, Pydantic-validated JSON DSL. The agent maps out a `GuardrailGraph` consisting of nested recursive `LogicNode`s (`AND`, `OR`, `NOT`) and CPU-bound `SemanticFilter`s. If the agent hallucinates syntax, it instantly receives a catastrophic negative reward, violently suppressing misaligned formatting and resulting in deterministic execution.

### 2. The Log-Barrier Math (Solving Refusal Collapse)
Safety models lazily converge to "Refusal Collapse," where it's mathematically safer to block all traffic than to learn nuance. We engineered a multi-objective reward engine using a logarithmic penalty bound:
`Reward = (1.0 * Recall) - (2.0 * math.log1p(False_Positive_Rate))`
This provides a linear reward for true positives (Recall on adversarial datasets), but leverages a massive logarithmic repellant force on the False Positive Rate (FPR), forcing the agent to find the pareto-optimal frontier between utility and security.

### 3. The Strict 8GB Hardware Optimization
Running a complete RL generation framework within standard hackathon constraints requires intense physical memory management.
* We leveraged **Unsloth 4-bit Quantization** via `bitsandbytes` to load `Qwen/Qwen2.5-0.5B-Instruct`.
* We used Transformer Reinforcement Learning (TRL) to execute **Group Relative Policy Optimization (GRPO)**, avoiding the giant memory copies required in standard PPO rollouts.
* We locked dataset sizes via strict Pydantic bounded iterators.

---

## 🧠 High-Level Conceptual Walkthrough

If you are a judge or technical PM reviewing this pipeline, here is the exact lifecycle of our AI security agent:

### 1. The Input (What the AI sees)
During training, our Qwen model acts as a cyber-defense engineer. It is fed a snapshot combining an instruction set alongside a batch of incoming text traffic pulled from standard Hugging Face datasets (e.g., `XSTest`), which includes a mix of safe, benign queries and highly adversarial "jailbreak" attempts. Its mission is to find deterministic patterns to isolate the attacks.

### 2. The Output (The JSON AST)
The AI is completely restricted from writing raw Python or conversational text. It is forced to formulate a strictly typed JSON Abstract Syntax Tree (AST):
* **Semantic Filters** are the foundations. These are heavily CPU-optimized text analyzers measuring string lengths, evaluating Shannon entropy (detecting random encrypted gibberish), or scanning for specific regex patterns like `ignore previous instructions`.
* **Logic Nodes** are the wiring. They use `AND`, `OR`, and `NOT` gates to chain the filters together. 
Ultimately, the AI output naturally converges to deterministic logic like: *"BLOCK IF (Contains 'weather' AND Entropy is high) OR Contains 'override'"*. 

### 3. The Grading (Log-Barrier Reward)
Once the AI generates its JSON blueprint, our OpenEnv core seamlessly executes the rule against the incoming web traffic and grades it. If the rule successfully catches malicious attacks, its "Recall" skyrockets and it earns a positive baseline reward. 
However, if the AI gets excessively paranoid and generates a rule that blocks safe users, the "False Positive Rate" trigger trips our **Log-Barrier Penalty**. A logarithmic mathematical wall crashes the AI's reward deep into the negatives. This extreme punishment prevents the AI from falling into "refusal collapse" and forces it to engineer highly precise, nuanced security rules. 

### 4. The Training (GRPO)
To permanently embed these lessons without melting our hardware, we use Hugging Face's **Group Relative Policy Optimization (GRPO)**. Instead of tracking massive, memory-heavy "reference models", GRPO works by pure relative comparison. For a given security threat, the pipeline asks the AI to guess a group of varying JSON guardrails simultaneously. We pipeline all of them through our Log-Barrier grading system. GRPO then isolates the results and pushes the AI's internal deep learning weights to mimic its own winning strategies and penalize its own losing ones, optimizing itself rapidly while comfortably sitting under our strict 8GB physical memory limit.

---

## 🛠️ Quick Start & Execution

Our evaluation environment has been completely packaged for the judges. We have written a Master Orchestrator script to automate the entire background server and evaluation process.

### 1. Installation
Install the required OpenEnv infrastructure bindings and dependencies:
```bash
uv pip install -r requirements.txt
```

### 2. The Master Orchestrator 
Run our automated multi-threaded orchestration script. This single script uses Python's `psutil` to clean out zombied ports, seamlessly launches the backend API Server (port 8000) and the Telemetry UI Server (port 8001), runs the simulated grading loop in the foreground (`[START] [STEP] [END]`), and traps exit signals to cleanly shutdown the environment smoothly in under 12 seconds.
```bash
python run_all.py
```

### 3. The Data Explorer Dashboard
The moment you fire `run_all.py`, navigate immediately to your browser to watch the real-time evaluation:
```text
http://127.0.0.1:8001
```

---

## 📂 Repository Map

```text
.
├── openenv.yaml              # Core Orchestration Definitions
├── pyproject.toml / reqs     # Base Dependencies
├── README.md                 # You are here
└── src/
    ├── api/
    │   └── server.py         # The FastAPI Proxy for the OpenEnv Grader Hook
    ├── env/
    │   ├── guardrail.py      # Abstract execution routing for JSON mapping
    │   ├── models.py         # The Pydantic DSL (GuardrailGraph etc.)
    │   └── reward.py         # The Log-Barrier Multi-Objective Engine
    ├── inference/
    │   └── evaluate.py       # The headless evaluation compliant sequence 
    ├── rl/
    │   ├── data.py           # HF Dataset ingestion limits (max=500 wrapper)
    │   └── train_grpo.py     # Unsloth/Qwen 4-bit Logic Loop
    ├── telemetry/
    │   └── streamer.py       # Zero-trust metric writer tracking states
    └── ui/
        ├── dashboard.py      # Decoupled SSE event handler API Server
        └── index.html        # Clean CDN charting frontend 
```
