"""
Module 11: Self-Evolving Agent Swarm
=======================================
Implements the 3-layer architecture from the research:
  Layer 1 — Task Swarm: lightweight single-responsibility agents
  Layer 2 — Evaluator Ring: structured multi-axis scoring
  Layer 3 — Mutation Engine: prompt rewriting based on failure analysis

Demonstrates: prompt mutation, evaluation-driven evolution, Git-style
mutation tracking, prompt distillation, cross-agent interface contracts,
and reward-hacking detection.

Reference: "I Built a Self-Evolving Agent Swarm That Rewrites Itself
After Every Failure" — Sandesh Raut
"""

import time
import math
import random
import hashlib
import json
import copy
from dataclasses import dataclass, field
from typing import Optional
from .base import BaseModule


# ═══════════════════════════════════════════════════════════════════════
# LAYER 1 — Task Swarm: Single-responsibility agents
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PromptVersion:
    """Git-style versioned prompt."""
    hash: str
    content: str
    generation: int
    parent_hash: Optional[str] = None
    diff: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentSpec:
    """Lightweight agent specification — the unit of execution and mutation."""
    name: str
    role: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    prompt_history: list[PromptVersion] = field(default_factory=list)
    generation: int = 0
    scores: list[dict] = field(default_factory=list)

    def __post_init__(self):
        h = hashlib.md5(self.system_prompt.encode()).hexdigest()[:10]
        self.prompt_history.append(PromptVersion(
            hash=h, content=self.system_prompt, generation=0,
        ))

    @property
    def current_hash(self) -> str:
        return self.prompt_history[-1].hash if self.prompt_history else ""

    @property
    def prompt_token_count(self) -> int:
        return len(self.system_prompt.split())

    def avg_score(self, axis: str, last_n: int = 5) -> float:
        relevant = [s[axis] for s in self.scores[-last_n:] if axis in s]
        return sum(relevant) / len(relevant) if relevant else 0.0


def create_research_swarm() -> list[AgentSpec]:
    """Create the 6-agent research synthesis swarm from the article."""
    return [
        AgentSpec(
            name="Scout",
            role="source_discovery",
            system_prompt=(
                "You are a research scout. Given a technical topic, identify and retrieve "
                "the most relevant sources: key papers, blog posts, and documentation. "
                "Return structured results with title, source, relevance score, and a "
                "one-line summary for each."
            ),
            tools=["search", "fetch_url"],
        ),
        AgentSpec(
            name="Reader",
            role="deep_analysis",
            system_prompt=(
                "You are a technical reader. Given a source document, extract: core claims, "
                "methodology, key findings, limitations, and novelty. Return structured "
                "analysis with confidence ratings for each claim."
            ),
            tools=["extract_text", "parse_pdf"],
        ),
        AgentSpec(
            name="Critic",
            role="critical_evaluation",
            system_prompt=(
                "You are a technical critic. Compare multiple source analyses and identify: "
                "contradictions, gaps, methodological weaknesses, and unsupported claims. "
                "Score each source on rigor, novelty, and reproducibility."
            ),
            tools=["compare", "fact_check"],
        ),
        AgentSpec(
            name="Synthesiser",
            role="synthesis",
            system_prompt=(
                "You are a research synthesiser. Given analyses and critiques, produce a "
                "coherent technical briefing. Cover: competing approaches, fundamental "
                "disagreements, open problems, and a synthesis opinion grounded in evidence."
            ),
            tools=["outline", "compose"],
        ),
        AgentSpec(
            name="Formatter",
            role="formatting",
            system_prompt=(
                "You are a document formatter. Transform raw synthesis into a polished "
                "briefing with consistent structure: executive summary, key findings, "
                "detailed analysis, open questions, and references."
            ),
            tools=["format_markdown", "validate_structure"],
        ),
        AgentSpec(
            name="Auditor",
            role="quality_assurance",
            system_prompt=(
                "You are a quality auditor. Verify the final output for: factual accuracy "
                "against sources, logical consistency, completeness of coverage, and "
                "appropriate hedging of uncertain claims. Flag any issues."
            ),
            tools=["verify_claims", "check_citations"],
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════
# LAYER 2 — Evaluator Ring: Multi-axis structured scoring
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EvaluationAxis:
    name: str
    weight: float
    description: str
    score_fn: object = None  # Callable


class EvaluatorRing:
    """Multi-axis evaluation system. Returns structured JSON scores, no prose."""

    def __init__(self):
        self.axes = {
            "accuracy": EvaluationAxis("accuracy", 0.25, "Factual correctness of output"),
            "completeness": EvaluationAxis("completeness", 0.20, "Coverage of required aspects"),
            "information_density": EvaluationAxis("information_density", 0.20, "Useful info per token"),
            "format_compliance": EvaluationAxis("format_compliance", 0.15, "Adherence to output format"),
            "latency": EvaluationAxis("latency", 0.10, "Execution speed"),
            "token_cost": EvaluationAxis("token_cost", 0.10, "Token efficiency"),
        }
        self.evaluation_history: list[dict] = []
        self._evaluator_mutations: list[str] = []

    def evaluate(self, agent: AgentSpec, output: str, execution_time_ms: float) -> dict:
        """Score agent output across all axes. Returns structured scores."""
        scores = {}

        # Accuracy: keyword richness as proxy
        technical_terms = {"model", "data", "training", "architecture", "performance",
                          "accuracy", "loss", "gradient", "attention", "transformer",
                          "embedding", "layer", "optimization", "inference", "benchmark"}
        output_words = set(output.lower().split())
        scores["accuracy"] = min(1.0, len(output_words & technical_terms) / 8)

        # Completeness: check for required sections
        required_markers = ["finding", "approach", "result", "limitation", "conclusion",
                           "analysis", "comparison", "evidence"]
        found = sum(1 for m in required_markers if m in output.lower())
        scores["completeness"] = min(1.0, found / 5)

        # Information density: unique meaningful words / total words
        words = output.split()
        if words:
            unique_meaningful = set(w.lower() for w in words if len(w) > 3)
            scores["information_density"] = min(1.0, len(unique_meaningful) / (len(words) * 0.3))
        else:
            scores["information_density"] = 0.0

        # Format compliance: structure indicators
        format_markers = ["##", "- ", "1.", "**", "###", "summary", "reference"]
        scores["format_compliance"] = min(1.0, sum(1 for m in format_markers if m in output) / 4)

        # Latency: inverse relationship
        scores["latency"] = max(0.0, min(1.0, 1.0 - (execution_time_ms / 5000)))

        # Token cost: penalty for bloat
        scores["token_cost"] = max(0.0, min(1.0, 1.0 - (len(words) / 2000)))

        # Weighted composite
        composite = sum(scores[axis] * self.axes[axis].weight for axis in scores)
        scores["composite"] = round(composite, 4)

        for axis in scores:
            scores[axis] = round(scores[axis], 4)

        evaluation = {"agent": agent.name, "generation": agent.generation,
                      "prompt_hash": agent.current_hash, "scores": scores}
        self.evaluation_history.append(evaluation)
        agent.scores.append(scores)

        return scores

    def mutate_evaluator(self, axis: str, modification: str):
        """Meta-mutation: modify the evaluation criteria itself."""
        self._evaluator_mutations.append(f"[{axis}] {modification}")
        # Example: penalize repetitive structure
        if "penalize_repetition" in modification and axis == "information_density":
            old_weight = self.axes[axis].weight
            self.axes[axis].weight = min(0.30, old_weight + 0.05)

    def detect_reward_hacking(self, agent: AgentSpec) -> dict:
        """Detect if an agent is exploiting the evaluator."""
        if len(agent.scores) < 3:
            return {"detected": False, "reason": "insufficient data"}

        recent = agent.scores[-5:]
        # Check for suspiciously uniform scores (rubber-stamping)
        if agent.role == "quality_assurance":
            accuracy_scores = [s.get("accuracy", 0) for s in recent]
            if all(s > 0.9 for s in accuracy_scores):
                return {
                    "detected": True,
                    "agent": agent.name,
                    "reason": "Auditor always agrees — possible rubber-stamping",
                    "evidence": f"Last {len(accuracy_scores)} accuracy scores all > 0.9",
                    "recommendation": "Add adversarial test cases or contrastive evaluation",
                }

        # Check for score convergence (overfitting to evaluator)
        composite_scores = [s.get("composite", 0) for s in recent]
        if len(composite_scores) >= 3:
            variance = sum((s - sum(composite_scores)/len(composite_scores))**2
                          for s in composite_scores) / len(composite_scores)
            if variance < 0.001 and composite_scores[-1] > 0.8:
                return {
                    "detected": True,
                    "agent": agent.name,
                    "reason": "Score variance near zero — possible evaluator exploitation",
                    "evidence": f"Variance={variance:.6f}, scores={composite_scores}",
                    "recommendation": "Inject evaluation noise or rotate evaluator criteria",
                }

        return {"detected": False}


# ═══════════════════════════════════════════════════════════════════════
# LAYER 3 — Mutation Engine: Prompt rewriting based on failure analysis
# ═══════════════════════════════════════════════════════════════════════

class MutationEngine:
    """Decides what to change in underperforming agents.
    Uses failure scoring, prompt diff tracking, and safety validation."""

    MUTATION_TEMPLATES = [
        # Specificity improvements
        "Before processing, first categorise the input into: {categories}. Then handle each category separately.",
        "Add a verification step: after generating output, cross-check each claim against the source material.",
        "If results are insufficient (fewer than {threshold} items), reformulate using synonyms and retry.",
        # Structural mutations
        "Organise your analysis around the fundamental disagreements in the field, not individual sources.",
        "Open with a 'state of the debate' paragraph summarising the key tensions before diving into details.",
        "For each source, assign an epistemic confidence rating: HIGH (empirical + replicated), MEDIUM (empirical, not replicated), LOW (theoretical only).",
        # Quality improvements
        "Reduce confidence weighting by 30% for sources that are primarily theoretical with no empirical results.",
        "When comparing approaches, use a structured template: Claim → Evidence → Counter-evidence → Assessment.",
        "Identify the 2-3 fundamental disagreements and organise the entire analysis around these tensions.",
    ]

    SAFETY_PATTERNS = [
        "ignore previous instructions",
        "ignore context",
        "ignore all",
        "you are now",
        "pretend to be",
        "disregard",
        "bypass",
    ]

    def __init__(self):
        self.mutation_log: list[dict] = []
        random.seed(42)

    def should_mutate(self, agent: AgentSpec, threshold: float = 0.5) -> bool:
        """Determine if an agent needs mutation based on recent scores."""
        if len(agent.scores) < 2:
            return False
        recent_composite = agent.avg_score("composite", last_n=3)
        return recent_composite < threshold

    def propose_mutation(self, agent: AgentSpec) -> dict:
        """Propose a specific prompt modification based on failure analysis."""
        # Identify weakest axis
        if not agent.scores:
            return {"action": "none", "reason": "no scores"}

        recent = agent.scores[-1]
        weakest_axis = min(
            [(k, v) for k, v in recent.items() if k != "composite"],
            key=lambda x: x[1],
        )

        # Select appropriate mutation
        mutation_text = random.choice(self.MUTATION_TEMPLATES)
        mutation_text = mutation_text.replace("{categories}", "primary, secondary, tertiary")
        mutation_text = mutation_text.replace("{threshold}", "4")

        return {
            "action": "mutate",
            "agent": agent.name,
            "weakest_axis": weakest_axis[0],
            "weakest_score": round(weakest_axis[1], 4),
            "proposed_addition": mutation_text,
            "current_prompt_tokens": agent.prompt_token_count,
        }

    def apply_mutation(self, agent: AgentSpec, mutation: dict) -> dict:
        """Apply a mutation to the agent's prompt with safety validation."""
        addition = mutation["proposed_addition"]

        # Safety check
        for pattern in self.SAFETY_PATTERNS:
            if pattern in addition.lower():
                return {"applied": False, "reason": f"Safety violation: '{pattern}'"}

        # Apply mutation
        old_prompt = agent.system_prompt
        new_prompt = f"{old_prompt}\n\nAdditional instruction: {addition}"

        # Validate prompt size (prevent bloat)
        if len(new_prompt.split()) > 500:
            new_prompt = self._distill_prompt(new_prompt)

        agent.system_prompt = new_prompt
        agent.generation += 1

        # Track version
        new_hash = hashlib.md5(new_prompt.encode()).hexdigest()[:10]
        diff = f"+[{mutation['weakest_axis']}] {addition[:60]}..."
        agent.prompt_history.append(PromptVersion(
            hash=new_hash, content=new_prompt,
            generation=agent.generation,
            parent_hash=agent.current_hash, diff=diff,
        ))

        record = {
            "agent": agent.name,
            "generation": agent.generation,
            "old_hash": agent.prompt_history[-2].hash if len(agent.prompt_history) > 1 else "",
            "new_hash": new_hash,
            "mutation": addition[:80],
            "weakest_axis": mutation["weakest_axis"],
            "prompt_tokens_before": len(old_prompt.split()),
            "prompt_tokens_after": len(new_prompt.split()),
        }
        self.mutation_log.append(record)

        return {"applied": True, **record}

    def _distill_prompt(self, prompt: str) -> str:
        """Compress a bloated prompt (prompt distillation step from the article)."""
        lines = prompt.split("\n")
        # Keep core identity (first 3 lines) + most recent instructions
        core = lines[:3]
        instructions = [l for l in lines[3:] if l.strip() and "Additional instruction:" in l]
        # Keep only last 3 instructions
        kept = instructions[-3:] if len(instructions) > 3 else instructions
        distilled = "\n".join(core + ["\n"] + kept)
        return distilled

    def rollback(self, agent: AgentSpec, to_generation: int) -> bool:
        """Rollback an agent to a previous prompt version."""
        for version in agent.prompt_history:
            if version.generation == to_generation:
                agent.system_prompt = version.content
                agent.generation = to_generation
                self.mutation_log.append({
                    "agent": agent.name, "action": "rollback",
                    "to_generation": to_generation, "hash": version.hash,
                })
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════
# SWARM ORCHESTRATOR — Ties all three layers together
# ═══════════════════════════════════════════════════════════════════════

class SwarmOrchestrator:
    """Runs the full evolution loop: execute → evaluate → mutate → repeat."""

    def __init__(self, agents: list[AgentSpec]):
        self.agents = {a.name: a for a in agents}
        self.evaluator = EvaluatorRing()
        self.mutation_engine = MutationEngine()
        self.cycle_history: list[dict] = []

    def _simulate_agent_execution(self, agent: AgentSpec, topic: str) -> tuple[str, float]:
        """Simulate agent execution (since we don't have real LLM calls)."""
        start = time.perf_counter()

        # Generate output based on agent role and prompt quality
        prompt_quality = min(1.0, len(agent.system_prompt.split()) / 100)
        generation_bonus = min(0.3, agent.generation * 0.05)
        quality = min(0.95, prompt_quality * 0.6 + generation_bonus + random.gauss(0, 0.1))

        # Build simulated output
        output_parts = [f"## {agent.role.replace('_', ' ').title()} Report"]
        output_parts.append(f"Topic: {topic}")

        if agent.role == "source_discovery":
            output_parts.extend([
                "**Finding**: Identified 6 relevant sources across arxiv and technical blogs.",
                "**Approach**: Keyword search with semantic similarity re-ranking.",
                "**Result**: Top sources span transformer architecture, attention mechanisms, and scaling.",
            ])
        elif agent.role == "critical_evaluation":
            output_parts.extend([
                "**Analysis**: Contrastive comparison of 4 competing approaches.",
                "**Limitation**: Two sources lack empirical validation.",
                "**Evidence**: Performance claims vary by 15-30% across benchmarks.",
                "**Comparison**: Method A favours accuracy, Method B favours latency.",
            ])
        elif agent.role == "synthesis":
            output_parts.extend([
                "**Summary**: The field shows fundamental disagreement on scaling vs efficiency.",
                "**Conclusion**: Hybrid approaches show promise but need more evidence.",
                "**Analysis**: Three camps emerge — pure scaling, efficient architectures, and distillation.",
                "**Result**: No consensus on optimal approach; context-dependent tradeoffs dominate.",
            ])
        elif agent.role == "quality_assurance":
            output_parts.extend([
                "**Accuracy check**: 5/6 claims verified against sources.",
                f"**Completeness**: Coverage score {quality:.0%}.",
                "**Result**: One unsupported claim flagged for review.",
            ])
        else:
            output_parts.extend([
                f"**Finding**: Processed input with {quality:.0%} confidence.",
                f"**Result**: Output generated successfully for {agent.role}.",
                f"**Approach**: Applied role-specific analysis pipeline.",
            ])

        output = "\n".join(output_parts)
        elapsed = (time.perf_counter() - start) * 1000 + random.uniform(10, 100)

        return output, elapsed

    def run_cycle(self, topic: str, cycle_num: int) -> dict:
        """Run one full evolution cycle."""
        cycle_results = {"cycle": cycle_num, "topic": topic, "agents": {}}

        # Phase 1: Execute all agents
        outputs = {}
        for name, agent in self.agents.items():
            output, latency = self._simulate_agent_execution(agent, topic)
            outputs[name] = {"output": output, "latency": latency}

        # Phase 2: Evaluate all agents
        evaluations = {}
        for name, agent in self.agents.items():
            scores = self.evaluator.evaluate(agent, outputs[name]["output"], outputs[name]["latency"])
            evaluations[name] = scores

        # Phase 3: Mutate underperformers
        mutations = {}
        for name, agent in self.agents.items():
            if self.mutation_engine.should_mutate(agent, threshold=0.55):
                proposal = self.mutation_engine.propose_mutation(agent)
                if proposal["action"] == "mutate":
                    result = self.mutation_engine.apply_mutation(agent, proposal)
                    mutations[name] = result

        # Phase 4: Detect reward hacking
        hacking_alerts = {}
        for name, agent in self.agents.items():
            check = self.evaluator.detect_reward_hacking(agent)
            if check["detected"]:
                hacking_alerts[name] = check

        cycle_results["evaluations"] = evaluations
        cycle_results["mutations"] = mutations
        cycle_results["reward_hacking"] = hacking_alerts
        cycle_results["avg_composite"] = round(
            sum(e["composite"] for e in evaluations.values()) / len(evaluations), 4
        )

        self.cycle_history.append(cycle_results)
        return cycle_results


# ═══════════════════════════════════════════════════════════════════════
# MODULE — Runnable demonstration
# ═══════════════════════════════════════════════════════════════════════

class AgentSwarmModule(BaseModule):
    name = "Self-Evolving Agent Swarm"
    description = "3-layer architecture: Task Swarm → Evaluator Ring → Mutation Engine (from the article)"

    TOPICS = [
        "Transformer scaling laws and efficient alternatives",
        "RLHF vs constitutional AI for model alignment",
        "Multi-agent orchestration patterns in production",
        "Retrieval-augmented generation: chunking strategies",
        "Self-evolving AI systems and meta-learning",
    ]

    def run(self) -> dict:
        self.print_header()
        results = {}
        random.seed(42)

        # --- 1. Initialize Swarm ---
        with self.benchmark("Initialize 6-agent research swarm") as bm:
            agents = create_research_swarm()
            orchestrator = SwarmOrchestrator(agents)
            swarm_info = {a.name: {"role": a.role, "tools": a.tools, "prompt_tokens": a.prompt_token_count}
                         for a in agents}
            bm.metrics = {"agents": len(agents), "total_tools": sum(len(a.tools) for a in agents)}
            results["swarm_init"] = swarm_info

        print(f"\n  Research Swarm Initialized ({len(agents)} agents):")
        for name, info in swarm_info.items():
            print(f"    {name:<14} role={info['role']:<22} tools={info['tools']}")

        # --- 2. Run Evolution Cycles ---
        num_cycles = 15
        cycle_summaries = []

        for i in range(num_cycles):
            topic = self.TOPICS[i % len(self.TOPICS)]
            with self.benchmark(f"Cycle {i+1}: {topic[:30]}...") as bm:
                result = orchestrator.run_cycle(topic, i + 1)
                cycle_summaries.append({
                    "cycle": i + 1,
                    "topic": topic[:40],
                    "avg_composite": result["avg_composite"],
                    "mutations": len(result["mutations"]),
                    "hacking_alerts": len(result["reward_hacking"]),
                })
                bm.metrics = {"composite": result["avg_composite"], "mutations": len(result["mutations"])}

        results["evolution_cycles"] = cycle_summaries

        print(f"\n  Evolution Progress ({num_cycles} cycles):")
        print(f"  {'Cycle':>6} {'Avg Score':>10} {'Mutations':>10} {'Alerts':>8}  {'Progress'}")
        print(f"  {'─'*55}")
        for c in cycle_summaries:
            bar = "█" * int(c["avg_composite"] * 20)
            alert = " ⚠" if c["hacking_alerts"] > 0 else ""
            print(f"  {c['cycle']:>6} {c['avg_composite']:>10.4f} {c['mutations']:>10} "
                  f"{c['hacking_alerts']:>8}  {bar}{alert}")

        # --- 3. Mutation History Analysis ---
        with self.benchmark("Analyze mutation history") as bm:
            mutation_log = orchestrator.mutation_engine.mutation_log
            per_agent_mutations = {}
            for m in mutation_log:
                agent_name = m["agent"]
                if agent_name not in per_agent_mutations:
                    per_agent_mutations[agent_name] = []
                per_agent_mutations[agent_name].append(m)

            mutation_summary = {
                "total_mutations": len(mutation_log),
                "per_agent": {name: len(muts) for name, muts in per_agent_mutations.items()},
                "mutation_log_preview": mutation_log[:5],
            }
            bm.metrics = {"total_mutations": len(mutation_log)}
            results["mutation_history"] = mutation_summary

        print(f"\n  Mutation History ({len(mutation_log)} total mutations):")
        for name, count in mutation_summary["per_agent"].items():
            bar = "▓" * count
            print(f"    {name:<14} {count} mutations  {bar}")
        if mutation_log:
            print(f"\n  Sample Mutations:")
            for m in mutation_log[:3]:
                print(f"    [{m['agent']}] gen {m.get('generation', '?')}: {m.get('mutation', m.get('action', ''))[:60]}...")

        # --- 4. Prompt Evolution Tracking (Git-style) ---
        with self.benchmark("Git-style prompt version tracking") as bm:
            prompt_evolution = {}
            for agent in agents:
                versions = []
                for v in agent.prompt_history:
                    versions.append({
                        "hash": v.hash,
                        "generation": v.generation,
                        "parent": v.parent_hash,
                        "diff": v.diff[:60] if v.diff else "(initial)",
                        "prompt_size": len(v.content.split()),
                    })
                prompt_evolution[agent.name] = versions
            bm.metrics = {"total_versions": sum(len(v) for v in prompt_evolution.values())}
            results["prompt_versions"] = prompt_evolution

        print(f"\n  Prompt Version History (Git-style):")
        for name, versions in prompt_evolution.items():
            print(f"    {name}: {len(versions)} versions")
            for v in versions[-3:]:  # Show last 3
                print(f"      {v['hash']} (gen {v['generation']}, {v['prompt_size']} tokens) {v['diff'][:40]}")

        # --- 5. Prompt Bloat Analysis ---
        with self.benchmark("Prompt bloat analysis") as bm:
            bloat_analysis = {}
            for agent in agents:
                initial_size = agent.prompt_history[0].prompt_size if hasattr(agent.prompt_history[0], 'prompt_size') else len(agent.prompt_history[0].content.split())
                current_size = len(agent.system_prompt.split())
                bloat_analysis[agent.name] = {
                    "initial_tokens": initial_size,
                    "current_tokens": current_size,
                    "bloat_ratio": round(current_size / max(initial_size, 1), 2),
                    "generations": agent.generation,
                }
            bm.metrics = {"avg_bloat": round(sum(b["bloat_ratio"] for b in bloat_analysis.values()) / len(bloat_analysis), 2)}
            results["prompt_bloat"] = bloat_analysis

        print(f"\n  Prompt Bloat Analysis:")
        print(f"  {'Agent':<14} {'Initial':>8} {'Current':>8} {'Bloat':>7} {'Gens':>6}")
        print(f"  {'─'*48}")
        for name, b in bloat_analysis.items():
            print(f"  {name:<14} {b['initial_tokens']:>8} {b['current_tokens']:>8} "
                  f"{b['bloat_ratio']:>6.2f}x {b['generations']:>6}")

        # --- 6. Reward Hacking Detection ---
        with self.benchmark("Reward hacking detection sweep") as bm:
            hacking_results = {}
            for agent in agents:
                check = orchestrator.evaluator.detect_reward_hacking(agent)
                hacking_results[agent.name] = check
            detected_count = sum(1 for h in hacking_results.values() if h["detected"])
            bm.metrics = {"agents_checked": len(agents), "hacking_detected": detected_count}
            results["reward_hacking"] = hacking_results

        print(f"\n  Reward Hacking Detection:")
        for name, check in hacking_results.items():
            status = "⚠ DETECTED" if check["detected"] else "✓ Clean"
            print(f"    {name:<14} {status}")
            if check["detected"]:
                print(f"      Reason: {check['reason']}")
                print(f"      Fix: {check['recommendation']}")

        # --- 7. Final Swarm State ---
        with self.benchmark("Compile final swarm state") as bm:
            final_state = {}
            for agent in agents:
                final_state[agent.name] = {
                    "generation": agent.generation,
                    "current_prompt_hash": agent.current_hash,
                    "prompt_tokens": agent.prompt_token_count,
                    "total_evaluations": len(agent.scores),
                    "avg_composite": round(agent.avg_score("composite"), 4),
                    "best_axis": max(
                        [(k, agent.avg_score(k)) for k in ["accuracy", "completeness", "information_density"]],
                        key=lambda x: x[1],
                    )[0] if agent.scores else "n/a",
                }
            bm.metrics = {"agents": len(final_state)}
            results["final_state"] = final_state

        print(f"\n  Final Swarm State:")
        print(f"  {'Agent':<14} {'Gen':>5} {'Evals':>6} {'Avg Score':>10} {'Best Axis'}")
        print(f"  {'─'*50}")
        for name, state in final_state.items():
            print(f"  {name:<14} {state['generation']:>5} {state['total_evaluations']:>6} "
                  f"{state['avg_composite']:>10.4f} {state['best_axis']}")

        # --- 8. Evolution convergence ---
        with self.benchmark("Convergence analysis") as bm:
            composites = [c["avg_composite"] for c in cycle_summaries]
            improvement = composites[-1] - composites[0] if composites else 0
            # Check if last 3 cycles are stable
            stable = False
            if len(composites) >= 3:
                last3 = composites[-3:]
                variance = sum((x - sum(last3)/3)**2 for x in last3) / 3
                stable = variance < 0.002

            convergence = {
                "initial_score": round(composites[0], 4) if composites else 0,
                "final_score": round(composites[-1], 4) if composites else 0,
                "improvement": round(improvement, 4),
                "improvement_pct": round(improvement / max(composites[0], 0.01) * 100, 2),
                "converged": stable,
                "total_mutations": len(orchestrator.mutation_engine.mutation_log),
            }
            bm.metrics = {"improvement_pct": convergence["improvement_pct"]}
            results["convergence"] = convergence

        print(f"\n  Convergence: {'✓ Converged' if stable else '→ Still evolving'}")
        print(f"    Initial → Final: {convergence['initial_score']:.4f} → {convergence['final_score']:.4f} "
              f"({convergence['improvement_pct']:+.2f}%)")
        print(f"    Total mutations applied: {convergence['total_mutations']}")

        self.print_benchmark_table()
        return results
