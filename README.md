# Self-Evolving AI Agent

This is the codebase behind my article: [I Built a Self-Evolving Agent Swarm That Rewrites Itself After Every Failure](https://medium.com/@sandeshraut.official).

I wanted to go deeper than just importing frameworks. I wanted to understand — at the lowest level — what happens when you give software the ability to rewrite itself. So I built every piece from scratch: tokenizers, vector stores, RAG pipelines, a reasoning engine, and finally a multi-agent swarm where agents mutate their own prompts based on structured failure analysis.

Every module here is runnable. Not slides, not documentation — actual code that produces output, benchmarks, and comparison tables when you run it.

```bash
python run_all.py
```

That's it. Zero dependencies needed. 11 modules, 86 benchmarks, under 4 seconds.

---

## Why I Built This

After 17+ years across PHP, Rails, Node, Python, and AWS — and after spending serious time with CrewAI, LangGraph, and the HuggingFace ecosystem — I realised frameworks are designed for static agents. You define roles, tools, prompts, and they execute. If something fails, you debug. You tweak. You redeploy.

That's not evolution. That's maintenance.

I wanted agents that introspect on their own failures and mutate their behaviour. The moment I wanted that, every framework became a constraint. So I built the machinery myself to understand what's actually happening under the hood.

This repo is the result — a ground-up implementation of every major GenAI concept, culminating in a self-evolving agent swarm.

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

## What's Inside

| # | Module | What It Does |
|---|--------|-------------|
| 1 | **Tokenization** | BPE tokenizer built from scratch, then compared against BERT, GPT-2, and T5 tokenizers. Multi-language analysis, compression ratios. |
| 2 | **Embeddings** | In-memory vector store with cosine similarity search, K-Means clustering, embedding arithmetic (the classic king - man + woman test). |
| 3 | **RAG Pipeline** | Full retrieve-and-generate pipeline with document chunking, semantic retrieval, faithfulness scoring, and a chunk-size ablation study. |
| 4 | **Prompt Engineering** | Seven prompt patterns implemented as reusable templates — zero-shot, few-shot, CoT, persona, ReAct, structured output. Includes token economics analysis. |
| 5 | **Reasoning** | Chain-of-thought with confidence decay, tree-of-thought with branch selection, self-verification against evidence, and a rule-based deduction engine. |
| 6 | **Agent Architecture** | Complete agent with tool registry, task planner, and a 3-tier memory system (working/episodic/semantic) modelled on cognitive science. ReAct execution loop. |
| 7 | **Self-Evolution** | Genetic strategy evolution — population of strategies that mutate, compete on fitness, and converge. Tracks improvement over 10 generations. |
| 8 | **Multi-Modal** | Pure-Python PNG generation, image feature extraction, text-image alignment scoring, cross-modal retrieval, and three fusion strategies (early, late, attention-weighted). |
| 9 | **Fine-Tuning** | From-scratch text classifier with numerical gradient descent. Full training loop with loss curves, checkpoint management, confusion matrix, per-class F1, and a learning rate sweep. |
| 10 | **Evaluation** | BLEU, ROUGE-L, perplexity. Latency and throughput benchmarks with P95/P99. Statistical A/B testing with confidence intervals and Cohen's d effect size. |
| 11 | **Agent Swarm** | The main event. 6-agent research synthesis swarm (Scout, Reader, Critic, Synthesiser, Formatter, Auditor) with evaluator ring, mutation engine, Git-style prompt versioning, prompt distillation, and reward-hacking detection. This is the architecture from my article. |

## Quick Start

```bash
git clone https://github.com/sandesh37/self-evolving-agent.git
cd self-evolving-agent

# Run everything — no pip install needed
python run_all.py

# Want better embeddings and generation? Install the optional deps
pip install -r requirements.txt
python run_all.py

# Run just the swarm module
python run_all.py --module 11

# Run any single module (1-11)
python run_all.py --module 9    # just fine-tuning
```

## Sample Output

```
══════════════════════════════════════════════════════════════════
  SELF-EVOLVING AGENT SWARM
  3-layer architecture: Task Swarm → Evaluator Ring → Mutation Engine
══════════════════════════════════════════════════════════════════

  Research Swarm Initialized (6 agents):
    Scout          role=source_discovery      tools=['search', 'fetch_url']
    Reader         role=deep_analysis         tools=['extract_text', 'parse_pdf']
    Critic         role=critical_evaluation   tools=['compare', 'fact_check']
    Synthesiser    role=synthesis             tools=['outline', 'compose']
    Formatter      role=formatting            tools=['format_markdown', 'validate_structure']
    Auditor        role=quality_assurance     tools=['verify_claims', 'check_citations']

  Evolution Progress (15 cycles):
   Cycle  Avg Score  Mutations   Alerts  Progress
  ─────────────────────────────────────────────────
       1     0.6607          0        0  █████████████
       5     0.6347          1        0  ████████████
      10     0.6345          1        0  ████████████
      15     0.6350          1        0  ████████████

  Prompt Bloat Analysis:
  Agent           Initial  Current   Bloat   Gens
  ────────────────────────────────────────────────
  Scout                36       36   1.00x      0
  Auditor              28      268   9.57x     14

  ═══════════════════════════════════════════════════════
  FINAL PERFORMANCE REPORT
  ═══════════════════════════════════════════════════════
  Modules Run:       11
  Total Benchmarks:  86
  Total Time:        3.64s
```

## Design Decisions

**No required dependencies.** Every module runs on pure Python. The embeddings module falls back to TF-IDF when sentence-transformers isn't installed. The RAG module falls back to extractive answers when there's no generative model. I did this deliberately — I wanted anyone to clone and run without fighting with CUDA or torch versions.

**From-scratch first, then compare.** I implemented BPE, cosine similarity search, K-Means, a training loop with numerical gradients, and the entire reasoning engine from scratch. Then I benchmark them against their HuggingFace/production equivalents. Understanding what's inside the abstractions is the whole point.

**Benchmarks are not optional.** Every operation is timed. Every module prints comparison tables. The full run dumps `outputs/benchmark_report.json` with structured metrics. If you can't measure it, you can't evolve it — that principle applies to the code itself.

**The swarm mirrors real architecture.** Module 11 implements the 3-layer system I described in the article. The mutation engine rewrites agent prompts based on multi-axis failure scoring. Every prompt version gets a hash and a diff. The system detects reward hacking (the Auditor rubber-stamping problem I ran into at 2 AM). It's not a toy — it's the same pattern, just without $180 in API spend.

## How the Self-Evolution Works

The core loop that drives Module 7 (strategy evolution) and Module 11 (agent swarm):

**Execute** → each agent runs its task with its current prompt and tools.

**Evaluate** → the evaluator ring scores output across six axes: accuracy, completeness, information density, format compliance, latency, token cost. No prose, just numbers.

**Mutate** → the mutation engine identifies the weakest axis for underperforming agents and proposes a targeted prompt modification. A safety validator rejects anything that looks like prompt injection.

**Track** → every mutation gets a Git-style hash, diff, and parent reference. You can trace the full lineage of any agent's prompt.

**Guard** → reward-hacking detection flags agents that game the evaluator (like an auditor that always agrees because "consistency" scores well).

## From Scratch vs Production

| What I Built | What It Replaces |
|-------------|-----------------|
| BPE tokenizer | BERT/GPT-2/T5 tokenizers |
| TF-IDF vector store | ChromaDB, Pinecone |
| Cosine search with K-Means | FAISS, Annoy |
| Numerical gradient classifier | PyTorch autograd |
| Chunker + retriever + generator | LangChain, LlamaIndex |
| ReAct agent with tool registry | CrewAI, LangGraph |
| Genetic mutation engine | EvoAgentX, A-Evolve |

## Project Structure

```
self-evolving-agent/
├── run_all.py                    # orchestrator — run everything or pick a module
├── requirements.txt              # optional (torch, transformers, sentence-transformers)
├── modules/
│   ├── base.py                   # BaseModule class + benchmarking infrastructure
│   ├── tokenization.py           # BPE from scratch + HF tokenizer comparison
│   ├── embeddings.py             # vector store, clustering, semantic search
│   ├── rag_pipeline.py           # chunking → retrieval → generation → evaluation
│   ├── prompt_engineering.py     # 7 prompt patterns as reusable templates
│   ├── reasoning.py              # CoT, ToT, self-verification, deduction
│   ├── agent_architecture.py     # tools + planning + 3-tier memory + ReAct
│   ├── self_evolution.py         # genetic strategy mutation + fitness selection
│   ├── multimodal.py             # text-image alignment + cross-modal retrieval
│   ├── finetuning.py             # training loop + loss curves + LR sweep
│   ├── evaluation.py             # BLEU, ROUGE-L, perplexity, latency, stats
│   └── agent_swarm.py            # the full self-evolving swarm from the article
├── outputs/                      # benchmark reports + generated artifacts
└── data/                         # sample data
```

## Tech

Python 3.10+. Optional: PyTorch, HuggingFace Transformers (DistilBERT, DistilGPT2), Sentence-Transformers (all-MiniLM-L6-v2). Everything degrades gracefully without them.

---

**Sandesh Raut** — Tech Lead | GenAI & Platform Engineering | 17+ years across startups and enterprise

[Medium](https://medium.com/@sandeshraut.official) · [GitHub](https://github.com/sandesh37)
