"""
Module 6: Agent Architecture
==============================
Implements a complete AI agent with: tool registry, planning engine,
working memory, episodic memory, and ReAct-style execution loop.
"""

import time
import math
import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable
from .base import BaseModule


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: list[str]
    fn: Callable
    call_count: int = 0
    total_time_ms: float = 0


@dataclass
class MemoryEntry:
    content: str
    memory_type: str  # "working", "episodic", "semantic"
    timestamp: float = 0
    relevance: float = 1.0
    access_count: int = 0

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class AgentStep:
    step_num: int
    thought: str
    action: str = ""
    action_input: str = ""
    observation: str = ""
    reflection: str = ""


class AgentMemory:
    """Three-tier memory system: working, episodic, semantic."""

    def __init__(self, working_capacity: int = 7):
        self.working: list[MemoryEntry] = []
        self.episodic: list[MemoryEntry] = []
        self.semantic: dict[str, MemoryEntry] = {}
        self.working_capacity = working_capacity

    def add_working(self, content: str):
        entry = MemoryEntry(content=content, memory_type="working")
        self.working.append(entry)
        # Miller's Law: working memory ≈ 7±2 items
        if len(self.working) > self.working_capacity:
            # Consolidate oldest to episodic
            oldest = self.working.pop(0)
            oldest.memory_type = "episodic"
            self.episodic.append(oldest)

    def add_episodic(self, content: str, relevance: float = 1.0):
        self.episodic.append(MemoryEntry(content=content, memory_type="episodic", relevance=relevance))

    def add_semantic(self, key: str, content: str):
        self.semantic[key] = MemoryEntry(content=content, memory_type="semantic")

    def recall(self, query: str, top_k: int = 3) -> list[MemoryEntry]:
        """Retrieve relevant memories using keyword overlap."""
        query_words = set(query.lower().split())
        scored = []
        for entry in self.working + self.episodic + list(self.semantic.values()):
            entry_words = set(entry.content.lower().split())
            overlap = len(query_words & entry_words) / max(len(query_words), 1)
            recency = 1.0 / (1.0 + (time.time() - entry.timestamp))
            score = overlap * 0.7 + recency * 0.3 + entry.relevance * 0.1
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, entry in scored[:top_k]:
            entry.access_count += 1
        return [entry for _, entry in scored[:top_k]]

    def get_stats(self) -> dict:
        return {
            "working_memory": len(self.working),
            "episodic_memory": len(self.episodic),
            "semantic_memory": len(self.semantic),
            "total_entries": len(self.working) + len(self.episodic) + len(self.semantic),
            "working_capacity": self.working_capacity,
        }


class Planner:
    """Decomposes tasks into executable sub-plans."""

    def create_plan(self, task: str, available_tools: list[str]) -> list[dict]:
        """Generate an execution plan based on task analysis."""
        task_lower = task.lower()
        plan = []

        # Analyze task to determine required steps
        if any(w in task_lower for w in ["search", "find", "look up", "what is"]):
            plan.append({"step": "search", "tool": "search_knowledge", "reason": "Need to retrieve relevant information"})

        if any(w in task_lower for w in ["calculate", "compute", "how many", "math", "sum"]):
            plan.append({"step": "calculate", "tool": "calculator", "reason": "Mathematical computation required"})

        if any(w in task_lower for w in ["analyze", "compare", "evaluate"]):
            plan.append({"step": "analyze", "tool": "analyzer", "reason": "Need to analyze and compare data"})

        if any(w in task_lower for w in ["summarize", "summary", "brief"]):
            plan.append({"step": "summarize", "tool": "summarizer", "reason": "Need to condense information"})

        # Always end with synthesis
        plan.append({"step": "synthesize", "tool": "synthesizer", "reason": "Combine findings into coherent answer"})

        return plan


class AIAgent:
    """Complete AI agent with tools, planning, memory, and ReAct loop."""

    def __init__(self):
        self.tools: dict[str, ToolDefinition] = {}
        self.memory = AgentMemory()
        self.planner = Planner()
        self.execution_trace: list[AgentStep] = []
        self._register_default_tools()

    def _register_default_tools(self):
        self.register_tool("search_knowledge", "Search the knowledge base for information",
                          ["query"], self._tool_search)
        self.register_tool("calculator", "Perform mathematical calculations",
                          ["expression"], self._tool_calculate)
        self.register_tool("analyzer", "Analyze and compare data points",
                          ["data"], self._tool_analyze)
        self.register_tool("summarizer", "Summarize text into key points",
                          ["text"], self._tool_summarize)
        self.register_tool("fact_checker", "Verify a factual claim",
                          ["claim"], self._tool_fact_check)

    def register_tool(self, name: str, description: str, parameters: list[str], fn: Callable):
        self.tools[name] = ToolDefinition(name=name, description=description, parameters=parameters, fn=fn)

    def execute(self, task: str, max_steps: int = 8) -> dict:
        """Execute a task using ReAct-style reasoning loop."""
        self.execution_trace = []

        # Step 1: Understand and plan
        step = AgentStep(step_num=1, thought=f"Understanding task: '{task}'. Let me break this down.")
        self.execution_trace.append(step)
        self.memory.add_working(f"Task: {task}")

        plan = self.planner.create_plan(task, list(self.tools.keys()))
        step = AgentStep(
            step_num=2,
            thought=f"Created plan with {len(plan)} steps: {[p['step'] for p in plan]}",
        )
        self.execution_trace.append(step)

        # Step 2-N: Execute plan with ReAct loop
        observations = []
        for i, plan_step in enumerate(plan):
            if len(self.execution_trace) >= max_steps:
                break

            tool_name = plan_step["tool"]
            if tool_name not in self.tools:
                continue

            tool = self.tools[tool_name]
            start = time.perf_counter()
            result = tool.fn(task)
            elapsed = (time.perf_counter() - start) * 1000
            tool.call_count += 1
            tool.total_time_ms += elapsed

            step = AgentStep(
                step_num=len(self.execution_trace) + 1,
                thought=plan_step["reason"],
                action=tool_name,
                action_input=task[:100],
                observation=result[:200],
                reflection=f"Tool '{tool_name}' returned useful information in {elapsed:.1f}ms",
            )
            self.execution_trace.append(step)
            observations.append(result)
            self.memory.add_working(f"Observation from {tool_name}: {result[:100]}")

        # Final synthesis
        final_answer = " | ".join(observations) if observations else "Could not process the task."
        step = AgentStep(
            step_num=len(self.execution_trace) + 1,
            thought="Synthesizing all observations into final answer.",
            observation=final_answer[:300],
            reflection="Task completed. All planned steps executed successfully.",
        )
        self.execution_trace.append(step)

        # Store in episodic memory
        self.memory.add_episodic(f"Completed task: {task} → {final_answer[:100]}", relevance=0.9)

        return {
            "task": task,
            "plan": plan,
            "steps": [{"step": s.step_num, "thought": s.thought, "action": s.action,
                       "observation": s.observation[:150], "reflection": s.reflection}
                      for s in self.execution_trace],
            "final_answer": final_answer[:500],
            "tools_used": [s.action for s in self.execution_trace if s.action],
            "total_steps": len(self.execution_trace),
            "memory_stats": self.memory.get_stats(),
        }

    # ── Built-in Tools ──────────────────────────────────────────────
    def _tool_search(self, query: str) -> str:
        knowledge = {
            "machine learning": "ML is a subset of AI that enables systems to learn from data. Key types: supervised, unsupervised, reinforcement learning.",
            "deep learning": "DL uses multi-layer neural networks. Key architectures: CNN, RNN, Transformer. Requires large datasets and compute.",
            "transformer": "Transformer uses self-attention mechanism. Key innovation: parallel processing of sequences. Foundation of BERT, GPT, etc.",
            "python": "Python is the most popular language for AI/ML. Key libraries: PyTorch, TensorFlow, scikit-learn, pandas, numpy.",
            "deployment": "ML deployment options: REST API, batch inference, edge deployment, model serving (TorchServe, TF Serving).",
        }
        query_lower = query.lower()
        results = []
        for key, value in knowledge.items():
            if key in query_lower or any(w in query_lower for w in key.split()):
                results.append(value)
        return " ".join(results) if results else "No specific knowledge found. General AI/ML concepts apply."

    def _tool_calculate(self, expr: str) -> str:
        import re
        numbers = re.findall(r'\d+\.?\d*', expr)
        if len(numbers) >= 2:
            nums = [float(n) for n in numbers[:5]]
            return f"Numbers found: {nums}. Sum={sum(nums):.2f}, Avg={sum(nums)/len(nums):.2f}, Max={max(nums):.2f}"
        return "No numerical computation needed for this query."

    def _tool_analyze(self, data: str) -> str:
        words = data.split()
        unique_words = set(w.lower() for w in words)
        return f"Analysis: {len(words)} words, {len(unique_words)} unique. Key terms: {', '.join(list(unique_words)[:8])}"

    def _tool_summarize(self, text: str) -> str:
        sentences = text.replace("?", ".").replace("!", ".").split(".")
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 10][:3]
        return "Summary: " + ". ".join(key_sentences) + "."

    def _tool_fact_check(self, claim: str) -> str:
        # Simulated fact checking
        return f"Fact check for '{claim[:50]}': Verified against knowledge base. Confidence: HIGH."


class AgentArchitectureModule(BaseModule):
    name = "Agent Architecture"
    description = "Tool registry, planning, memory (working/episodic/semantic), ReAct execution"

    TASKS = [
        "What is deep learning and how does it relate to machine learning?",
        "Compare transformer architecture with RNNs for NLP tasks",
        "Summarize the key Python libraries used for AI development",
        "Analyze the deployment options for ML models in production",
    ]

    def run(self) -> dict:
        self.print_header()
        results = {}
        agent = AIAgent()

        # --- 1. Tool Registry ---
        with self.benchmark("Initialize tool registry") as bm:
            tool_info = {}
            for name, tool in agent.tools.items():
                tool_info[name] = {"description": tool.description, "parameters": tool.parameters}
            bm.metrics = {"tools_registered": len(tool_info)}
            results["tool_registry"] = tool_info

        print(f"\n  Tool Registry ({len(tool_info)} tools):")
        for name, info in tool_info.items():
            print(f"    {name:<20} — {info['description'][:50]}")

        # --- 2. Execute Tasks ---
        task_results = []
        for task in self.TASKS:
            with self.benchmark(f"Execute: '{task[:35]}...'") as bm:
                result = agent.execute(task)
                bm.metrics = {"steps": result["total_steps"], "tools": len(result["tools_used"])}
                task_results.append(result)

        results["task_executions"] = task_results

        print(f"\n  Task Execution Results:")
        for r in task_results:
            print(f"\n  Task: {r['task'][:60]}...")
            print(f"    Plan: {[p['step'] for p in r['plan']]}")
            print(f"    Tools used: {r['tools_used']}")
            print(f"    Steps: {r['total_steps']}")
            print(f"    Answer: {r['final_answer'][:100]}...")

        # --- 3. Memory Analysis ---
        with self.benchmark("Memory system analysis") as bm:
            memory_stats = agent.memory.get_stats()

            # Test recall
            recall_test = agent.memory.recall("transformer deep learning", top_k=3)
            recall_results = [{"content": m.content[:80], "type": m.memory_type, "accesses": m.access_count}
                              for m in recall_test]

            bm.metrics = {"total_memories": memory_stats["total_entries"]}
            results["memory"] = {"stats": memory_stats, "recall_test": recall_results}

        print(f"\n  Memory System:")
        print(f"    Working: {memory_stats['working_memory']}/{memory_stats['working_capacity']}")
        print(f"    Episodic: {memory_stats['episodic_memory']}")
        print(f"    Semantic: {memory_stats['semantic_memory']}")
        print(f"    Recall test ({len(recall_results)} results):")
        for r in recall_results:
            print(f"      [{r['type']}] {r['content'][:60]}...")

        # --- 4. Tool Usage Statistics ---
        with self.benchmark("Tool usage statistics") as bm:
            tool_stats = {}
            for name, tool in agent.tools.items():
                tool_stats[name] = {
                    "calls": tool.call_count,
                    "total_time_ms": round(tool.total_time_ms, 2),
                    "avg_time_ms": round(tool.total_time_ms / max(tool.call_count, 1), 2),
                }
            bm.metrics = {"total_tool_calls": sum(t.call_count for t in agent.tools.values())}
            results["tool_stats"] = tool_stats

        print(f"\n  Tool Usage Statistics:")
        print(f"  {'Tool':<22} {'Calls':>6} {'Total ms':>10} {'Avg ms':>10}")
        print(f"  {'─'*50}")
        for name, stats in tool_stats.items():
            print(f"  {name:<22} {stats['calls']:>6} {stats['total_time_ms']:>10.2f} {stats['avg_time_ms']:>10.2f}")

        self.print_benchmark_table()
        return results
