# Self-Evolving AI Agent

A comprehensive, **runnable** proof-of-concept covering 11 core GenAI concepts — from tokenization to a self-evolving agent swarm that rewrites its own prompts after every failure.

Built as a portfolio piece to demonstrate deep, hands-on understanding of the full AI/ML stack. Every module produces real output, benchmarks, and comparison tables when you run it.

> *"There's a moment in every serious GenAI project where you stop being the architect and start being the observer."*
>
> — From the companion article: [I Built a Self-Evolving Agent Swarm That Rewrites Itself After Every Failure](https://medium.com/@sandeshraut.official)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      run_all.py (Orchestrator)                  │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐  │
│  │Tokenizer │ │Embeddings│ │   RAG    │ │ Prompt Engineering│  │
│  │  BPE     │ │ VectorDB │ │ Pipeline │ │ Zero/Few/CoT      │  │
│  │  HF      │ │ K-Means  │ │ Chunking │ │ Persona/ReAct     │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────────┘  │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │Reasoning │ │  Agent   │ │  Self-   │ │   Multi-Modal    │   │
│  │  CoT     │ │  Tools   │ │Evolution │ │  Text + Image    │   │
│  │  ToT     │ │  Memory  │ │  Fitness │ │  Cross-Modal     │   │
│  │  Verify  │ │  Plan    │ │  Mutate  │ │  Fusion          │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌────────────────────────────────┐  │
│  │Fine-Tune │ │Evaluation│ │     AGENT SWARM (Module 11)    │  │
│  │ Train    │ │ BLEU     │ │  ┌───────┐  ┌──────────────┐   │  │
│  │ Loss     │ │ ROUGE-L  │ │  │ Task  │→ │  Evaluator   │   │  │
│  │ Metrics  │ │ Latency  │ │  │ Swarm │  │  Ring         │   │  │
│  │ LR Study │ │ Stats    │ │  └───────┘  └──────┬───────┘   │  │
│  └──────────┘ └──────────┘ │       ↑            │            │  │
│                             │       └── Mutation ←┘            │  │
│                             │           Engine                 │  │
│                             └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## What's Covered

| # | Module | GenAI Concept | Key Implementations |
|---|--------|--------------|---------------------|
| 1 | Tokenization | Text encoding | BPE from scratch, HuggingFace tokenizer comparison, multi-language analysis |
| 2 | Embeddings | Vector representations | Sentence embeddings, cosine similarity, vector DB, K-Means clustering, embedding arithmetic |
| 3 | RAG Pipeline | Retrieval-augmented generation | Document chunking, semantic retrieval, generation, faithfulness scoring, chunk-size ablation |
| 4 | Prompt Engineering | Prompt design patterns | Zero-shot, few-shot, chain-of-thought, persona, ReAct, structured output, token economics |
| 5 | Reasoning | Structured thinking | Chain-of-thought, tree-of-thought, self-verification, multi-step logical deduction |
| 6 | Agent Architecture | Agentic systems | Tool registry, planning engine, 3-tier memory (working/episodic/semantic), ReAct loop |
| 7 | Self-Evolution | Self-improvement | Strategy mutation, fitness evaluation, genetic selection, convergence analysis |
| 8 | Multi-Modal | Text + image | Synthetic image generation, feature extraction, text-image alignment, cross-modal retrieval, fusion strategies |
| 9 | Fine-Tuning | Model training | From-scratch classifier, training loop, loss curves, checkpoints, precision/recall/F1, LR sweep |
| 10 | Evaluation | Benchmarking | BLEU, ROUGE-L, perplexity, latency/throughput benchmarks, statistical A/B testing (Cohen's d) |
| 11 | **Agent Swarm** | **Self-evolving multi-agent** | **6-agent research swarm, evaluator ring, mutation engine, Git-style prompt tracking, reward-hacking detection, prompt distillation** |

## Quick Start

```bash
# Clone
git clone https://github.com/sandesh37/self-evolving-agent.git
cd self-evolving-agent

# Run everything (zero dependencies needed — pure Python fallbacks for all modules)
python run_all.py

# Run with HuggingFace models for enhanced output
pip install -r requirements.txt
python run_all.py

# Run a single module
python run_all.py --module 11    # Just the agent swarm
python run_all.py --module 1     # Just tokenization

# Fast mode (skip large model downloads)
python run_all.py --fast
```

## Sample Output

```
╔══════════════════════════════════════════════════════════════════╗
║            🧠  SELF-EVOLVING AI AGENT  🧠                       ║
║            Covering 11 Core AI/ML Concepts                      ║
╚══════════════════════════════════════════════════════════════════╝

▶▶▶ Running Module 11/11: Agent Swarm

  Research Swarm Initialized (6 agents):
    Scout          role=source_discovery      tools=['search', 'fetch_url']
    Reader         role=deep_analysis         tools=['extract_text', 'parse_pdf']
    Critic         role=critical_evaluation   tools=['compare', 'fact_check']
    Synthesiser    role=synthesis             tools=['outline', 'compose']
    Formatter      role=formatting            tools=['format_markdown', 'validate_structure']
    Auditor        role=quality_assurance     tools=['verify_claims', 'check_citations']

  Evolution Progress (15 cycles):
  Cycle  Avg Score  Mutations   Alerts  Progress
  ─────────────────────────────────────────────
      1     0.3842          0        0  ███████
      5     0.4915          3        0  █████████
     10     0.5234          2        0  ██████████
     15     0.5512          1        1  ███████████ ⚠

  Reward Hacking Detection:
    Auditor        ⚠ DETECTED
      Reason: Auditor always agrees — possible rubber-stamping
      Fix: Add adversarial test cases or contrastive evaluation

═══════════════════════════════════════════════════════════════════
  FINAL PERFORMANCE REPORT
═══════════════════════════════════════════════════════════════════
  Modules Run:       11
  Total Benchmarks:  85+
  Total Time:        <30s (pure Python) | ~2min (with HF models)
```

## Design Decisions

**Zero required dependencies.** Every module runs with pure Python. HuggingFace models are optional enhancements — the system gracefully falls back to TF-IDF embeddings, keyword classifiers, and extractive generation when they're unavailable.

**Benchmarks baked in.** Every operation is timed. Every module prints comparison tables. The `outputs/benchmark_report.json` captures everything for later analysis.

**From-scratch implementations alongside production tools.** The BPE tokenizer, vector store, K-Means clustering, text classifier, and reasoning engine are all implemented from scratch — then compared with their HuggingFace equivalents. This demonstrates understanding of what's *inside* the abstractions.

**The swarm module mirrors a real system.** The 3-layer architecture (Task Swarm → Evaluator Ring → Mutation Engine) is the same pattern described in the companion article, implementing prompt mutation, Git-style version tracking, prompt distillation, and reward-hacking detection.

## Project Structure

```
self-evolving-agent/
├── run_all.py                    # Master orchestrator
├── requirements.txt              # Optional deps (torch, transformers)
├── README.md
├── modules/
│   ├── base.py                   # BaseModule + BenchmarkResult
│   ├── tokenization.py           # Module 1: BPE, HF tokenizers
│   ├── embeddings.py             # Module 2: Embeddings, vector search
│   ├── rag_pipeline.py           # Module 3: RAG with evaluation
│   ├── prompt_engineering.py     # Module 4: Prompt patterns
│   ├── reasoning.py              # Module 5: CoT, ToT, verification
│   ├── agent_architecture.py     # Module 6: Agent + tools + memory
│   ├── self_evolution.py         # Module 7: Genetic strategy evolution
│   ├── multimodal.py             # Module 8: Text-image processing
│   ├── finetuning.py             # Module 9: Training loop + metrics
│   ├── evaluation.py             # Module 10: Benchmarking framework
│   └── agent_swarm.py            # Module 11: Self-evolving agent swarm
├── outputs/                      # Generated artifacts + benchmark report
└── data/                         # Sample data files
```

## Tech Stack

- **Python 3.10+** — zero external dependencies for core functionality
- **PyTorch** (optional) — tensor operations for model modules
- **HuggingFace Transformers** (optional) — pre-trained models (DistilBERT, DistilGPT2)
- **Sentence-Transformers** (optional) — text embeddings (all-MiniLM-L6-v2)

## Key Concepts Demonstrated

### Self-Evolution (The Core Innovation)
The system implements a feedback loop where agents improve autonomously:

1. **Execute** — Agents run their assigned tasks
2. **Evaluate** — Multi-axis scoring (accuracy, completeness, information density, latency, cost)
3. **Mutate** — Underperformers get prompt rewrites targeting their weakest axis
4. **Track** — Every mutation is Git-tagged with hash, diff, and parent reference
5. **Guard** — Safety validation rejects prompt injections; reward-hacking detection flags exploitative agents

### From-Scratch vs Production
| Component | From Scratch | Production Equivalent |
|-----------|-------------|----------------------|
| Tokenizer | BPE implementation | BERT/GPT-2/T5 tokenizers |
| Embeddings | TF-IDF vectors | Sentence-Transformers |
| Vector DB | In-memory cosine search | ChromaDB / Pinecone |
| Classifier | Numerical gradient descent | PyTorch autograd |
| RAG | Custom chunker + retriever | LangChain / LlamaIndex |
| Agent | ReAct loop with tool registry | CrewAI / LangGraph |
| Evolution | Genetic mutation engine | EvoAgentX / A-Evolve |

---

**Author:** Sandesh Raut — Tech Lead, GenAI & Platform Engineering
**Article:** [I Built a Self-Evolving Agent Swarm That Rewrites Itself After Every Failure](https://medium.com/@sandeshraut.official)
