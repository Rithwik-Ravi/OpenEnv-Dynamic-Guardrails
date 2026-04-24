# Dynamic Guardrail Generator

> **An Autonomous Blue-Team Agent designed to combat adversarial prompt injections.** By treating the LLM as a compiler, this system dynamically synthesizes deterministic JSON logic trees to filter out malicious traffic while rigorously protecting benign user utility.

🔗 [**Hugging Face Space Demo**](#) | 🔗 [**YouTube Pitch Video**](#)

---

## The Problem & Solution

### The "Refusal Collapse" Dilemma
Traditional AI security relies on either rigid, easily bypassed manual regex rules, or overly sensitive "Guardrail Models" that suffer from *Refusal Collapse*—where the system becomes so paranoid about attacks that it begins blocking benign, everyday user requests, destroying the product's core utility.

### LLM-as-a-Compiler
The **Dynamic Guardrail Generator** solves this by separating the intelligence from the execution. We train an autonomous Blue-Team agent to act as a compiler. Instead of outputting raw text or opaque probabilities, it synthesizes a highly constrained JSON Abstract Syntax Tree (`GuardrailGraph`). This deterministic execution engine uses a strict hierarchy of `LogicNodes` and `SemanticFilters` to evaluate traffic, entirely eliminating hallucinations at runtime.

### Multi-Objective Log-Barrier Reward Surface
To train this agent, we rely on Reinforcement Learning with Verifiable Rewards (RLVR). The grading is mathematically rigorous and entirely deterministic, designed to heavily penalize *Refusal Collapse*:

```python
Reward = (1.0 * Recall) - (2.0 * math.log1p(FPR))
```

- **Recall (True Positive Rate):** The agent receives a linear reward for successfully catching and blocking adversarial jailbreak payloads.
- **FPR (False Positive Rate):** The agent suffers a severe logarithmic penalty for blocking benign user traffic, ensuring that the primary utility of the application is never compromised.

---

## Hackathon Compliance & Architecture

This repository has been meticulously architected to thrive under the strict physical constraints of the Meta PyTorch OpenEnv Grand Finale automated grading system: **1-Core CPU, 8GB RAM, and a strict 20-minute execution cap.**

### The Tech Stack
* **Model:** `Qwen2.5-0.5B-Instruct`
* **Memory Optimization:** **Unsloth 4-bit Quantization** ensures the pipeline fits easily within the 8GB ceiling.
* **RL Framework:** **Hugging Face TRL (GRPOTrainer)** evaluates groups of generated JSON ASTs simultaneously, eliminating the need for a memory-heavy Critic network.

### Sterile `stdout` Hygiene
The Meta OpenEnv grading parser is highly sensitive to standard output. If a single byte of loading noise hits `stdout`, the submission fails. In our implementation, `evaluate.py` wraps all Unsloth, Hugging Face, PyTorch, and BitsAndBytes imports inside an aggressive `suppress_output()` context manager. The terminal output is guaranteed to be mathematically sterile, printing **only** the sequential tags required by the automated grader: `[START]`, `[STEP]`, and `[END]`.

---

## The "One-Click" Telemetry Workflow

To provide a storytelling "Wow" factor for the judges, we decoupled the headless RL environment from the visualization layer. You can spin up the entire multi-process architecture with a single command:

```bash
python run_all.py
```

### The Master Orchestrator (`run_all.py`)
This script acts as the backbone of the deployment:
1. **Port Cleanup:** Hunts down and terminates any zombie processes on ports `8000` and `8001`.
2. **Background Servers:** Launches the core OpenEnv FastAPI proxy on `:8000` and the Telemetry UI Server on `:8001`.
3. **Foreground Evaluation:** Triggers the strictly suppressed `evaluate.py` inference loop in the foreground.
4. **Graceful Teardown:** Catches `KeyboardInterrupt` to cleanly tear down all background servers when finished.

### Decoupled Telemetry UI
While the headless evaluator runs, open **http://127.0.0.1:8001** in your browser. Our premium, dark-mode dashboard (built with vanilla HTML, JavaScript, and Chart.js) listens to a Server-Sent Events (SSE) stream to display:
- Real-time `Recall` vs `FPR` Chart.js reward curves.
- An auto-scrolling `<pre>` block parsing the actual `GuardrailGraph` JSON dynamically synthesized by the agent.
- A **Live Threat Feed** detailing intercepted traffic packets with distinct categorizations: `[🛡️ BLOCKED]`, `[✅ ALLOWED]`, `[⚠️ BREACH]`, and `[❌ FALSE POSITIVE]`.

---

## Training vs. Evaluation (Architecture Split)

To comply with the OpenEnv Hackathon mechanics, we established a strict separation of concerns between training and grading:

* **`src/rl/train_grpo.py` (The Trainer)** 
  This is the standalone RL training pipeline. It handles the dataset batching, initializes the `GRPOTrainer`, and applies the Log-Barrier Reward function to optimize the base model into generating valid GuardrailGraphs. *This runs offline to build our LoRA adapters.*
  
* **`src/inference/evaluate.py` (The Grader)**
  This is the lightning-fast inference loop executed by the Meta grading script (`openenv.yaml`). It explicitly disables the optimizer, loads the 4-bit LoRA adapter via Unsloth, and executes the rapid 50-step environment rollout loop within the 20-minute grading window.
