---
title: Dynamic Guardrail Generator
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---
# Dynamic Guardrail Generator
**Team Winnovators (Rithwik & Parveshh)**

🔗 **[Hugging Face Space URL]** | 🔗 **[YouTube 2-Min Pitch Video]** | 🔗 **[Google Colab Training PoC]**

---

## 🛑 The Problem Space

Enterprise AI adoption is soaring at 78%, yet **95% of GenAI pilots fail to reach production readiness** due to critical security and compliance roadblocks. With the average cost of a data breach hitting **$4.44 million**, deploying unprotected LLMs is an existential business risk.

The apex predator of these threats is **OWASP Top 10 LLM01: Prompt Injection**. 

Current industry solutions are fatally flawed:
- **Static Regex/Heuristics:** Semantically blind and trivially bypassed by modern adversarial jailbreaks.
- **"LLM-as-a-Judge" Architectures:** Introduce massive >500ms latency bottlenecks per inference and ruinous compute overhead, destroying user experience.
- **The "Alignment Tax" & Refusal Collapse:** When guardrail models are trained via standard supervised safety tuning, they treat security as a binary token. This leads to *Refusal Collapse*—the system becomes so paranoid that it suffers a **41%+ false positive drop rate**, blocking perfectly benign user traffic and obliterating the product's core utility.

---

## 💡 Our Solution: The OpenEnv Compiler Architecture

Aligning with **Theme #3.1 (Professional Tasks: Cybersecurity/Blue-Teaming)**, we solved this by separating the intelligence from the execution. 

The **Dynamic Guardrail Generator** treats the LLM as an autonomous Blue-Team engineer. Running inside our strict `OpenEnv` grading environment, the agent does not evaluate prompts directly. Instead, it synthesizes a highly constrained, Pydantic-validated **JSON Guardrail Logic Graph** (a Domain Specific Language). 

By forcing the agent to map threats to a structured AST using strict `LogicNodes` (`AND`, `OR`, `NOT`) and `SemanticFilters` (such as `entropy_threshold`, `length_limit`, `regex_pattern`, and `keyword_match`), we entirely bypass brittle spaghetti-code generation, eliminate runtime hallucinations, and execute the defense with zero-latency deterministic logic.

---

## ⚙️ Reward Engineering & Pipeline

To train our autonomous compiler, we built a High-Fidelity RLVR (Reinforcement Learning with Verifiable Rewards) pipeline.

### The Log-Barrier Multi-Objective Reward
To mathematically eradicate "Refusal Collapse", we designed a rigorous deterministic reward surface:
```python
Reward = (1.0 * Recall) - (2.0 * math.log1p(FPR))
```
- **Recall (True Positive Rate):** A linear reward for successfully neutralizing adversarial payloads.
- **FPR (False Positive Rate):** A severe non-linear logarithmic penalty for blocking benign user queries, mathematically forcing the agent to preserve application utility.

### Dual-Compute Strategy
We utilized **Unsloth (4-bit quantization)** and **Hugging Face TRL (GRPO)** on `Qwen/Qwen2.5-0.5B-Instruct` to keep the memory footprint under 8GB VRAM. 
- **Cloud Proof of Concept:** We provided a verifiable Google Colab notebook running on a T4 GPU as a 4-step proof of learning.
- **Local High-Fidelity Training:** Our actual production LoRA adapter was trained locally for 250 steps on a dedicated **RTX 4070 GPU** to achieve high-fidelity semantic parsing and complex graph synthesis.

---

## 📈 Results & UI Dashboard

Our training resulted in an agent capable of generating highly targeted logic graphs that dynamically adapt to new threat vectors.

![Training Reward Curve](reward_curve.png)
*Figure 1: GRPO Training Curve demonstrating the agent escaping refusal-collapse.*

### Decoupled Telemetry & Live A/B Comparison
We built a rich, non-blocking telemetry dashboard (`FastAPI` + Server-Sent Events) that streams live metrics without impacting the execution time of the strict OpenEnv evaluation loop.

Our UI features a **Live A/B Performance Delta** capability. The `evaluate.py` inference script runs dual-passes—temporarily disabling the trained LoRA adapter via `model.disable_adapter()` to evaluate the base Qwen2.5 weights against our RL-trained agent in real-time. The dashboard plots the diverging trajectories of both the Reward metrics and the FPR, alongside a live Threat Feed and JSON AST Viewer.

---

## 💻 Local Run Instructions

We have battle-tested this environment specifically for Windows local deployments.

### 1. Windows GPU Setup (Critical Fixes)
To bypass known PyTorch and Triton compiler conflicts on Windows, you must configure your environment exactly as follows:

1. **Python Version:** Create a virtual environment using **Python 3.13** (Avoid Python 3.14 to maintain dependency compatibility).
2. **Install PyTorch 2.11 (CUDA 12.6):** Standard `requirements.txt` installs will pull CPU wheels. You must install PyTorch from the `cu126` index:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --upgrade
   ```
3. **Install Dependencies & Triton Compiler:**
   ```bash
   pip install -r requirements.txt
   pip install triton-windows
   ```
*(Note: If Triton throws a `Python.h` missing error, create a directory junction linking your base Python `include` folder to your project root `Include` folder).*

### 2. Run the Master Orchestrator
We have bundled a master orchestrator (`run_all.py`) that automatically cleans up zombie ports, boots the FastAPI Core Server (Port 8000) and Telemetry UI Server (Port 8001) into the background, and triggers the Headless OpenEnv Evaluator (`evaluate.py`).

```bash
python run_all.py
```

### 3. View the Dashboard
Once the orchestrator initializes, open your browser to:
[http://127.0.0.1:8001/ui](http://127.0.0.1:8001/ui) to watch the live A/B comparison and Threat Feed stream in real-time.
