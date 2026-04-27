"""
Module 5: Chain-of-Thought & Reasoning Engine
================================================
Implements structured reasoning: CoT, tree-of-thought, self-verification,
and multi-step logical deduction with trace visualization.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from .base import BaseModule


@dataclass
class ThoughtNode:
    """A single node in a reasoning tree."""
    id: int
    content: str
    confidence: float
    parent_id: Optional[int] = None
    children_ids: list[int] = field(default_factory=list)
    is_verified: bool = False
    verification_note: str = ""


class ReasoningEngine:
    """Implements multiple reasoning strategies."""

    def __init__(self):
        self.thought_counter = 0
        self.thoughts: dict[int, ThoughtNode] = {}

    def _new_thought(self, content: str, confidence: float, parent_id: Optional[int] = None) -> ThoughtNode:
        self.thought_counter += 1
        node = ThoughtNode(
            id=self.thought_counter, content=content,
            confidence=confidence, parent_id=parent_id,
        )
        self.thoughts[node.id] = node
        if parent_id and parent_id in self.thoughts:
            self.thoughts[parent_id].children_ids.append(node.id)
        return node

    def chain_of_thought(self, problem: str, steps: list[str]) -> dict:
        """Linear chain-of-thought reasoning."""
        chain = []
        parent_id = None
        cumulative_conf = 1.0

        for i, step in enumerate(steps):
            conf = max(0.7, 1.0 - i * 0.05)  # Confidence decays slightly
            cumulative_conf *= conf
            node = self._new_thought(step, confidence=conf, parent_id=parent_id)
            chain.append({
                "step": i + 1,
                "thought": step,
                "confidence": round(conf, 3),
                "cumulative_confidence": round(cumulative_conf, 3),
            })
            parent_id = node.id

        return {
            "strategy": "chain_of_thought",
            "problem": problem,
            "chain": chain,
            "final_confidence": round(cumulative_conf, 3),
            "total_steps": len(chain),
        }

    def tree_of_thought(self, problem: str, branches: list[list[str]], evaluate: bool = True) -> dict:
        """Tree-of-thought: explore multiple reasoning paths, pick best."""
        paths = []
        root = self._new_thought(f"Problem: {problem}", confidence=1.0)

        for branch_idx, branch_steps in enumerate(branches):
            parent_id = root.id
            path = {"branch": branch_idx + 1, "steps": [], "confidence": 1.0}
            cumulative = 1.0

            for step in branch_steps:
                conf = max(0.6, 1.0 - len(step) * 0.002)  # Longer steps = more complex = slightly less confident
                cumulative *= conf
                node = self._new_thought(step, confidence=conf, parent_id=parent_id)
                path["steps"].append({"thought": step, "confidence": round(conf, 3)})
                parent_id = node.id

            path["confidence"] = round(cumulative, 3)
            paths.append(path)

        # Evaluate and select best path
        if evaluate:
            paths.sort(key=lambda p: p["confidence"], reverse=True)
            best = paths[0]
        else:
            best = paths[0]

        return {
            "strategy": "tree_of_thought",
            "problem": problem,
            "paths_explored": len(paths),
            "all_paths": paths,
            "selected_path": best["branch"],
            "best_confidence": best["confidence"],
        }

    def self_verify(self, claim: str, evidence: list[str], counter_evidence: list[str] = None) -> dict:
        """Self-verification: check a claim against evidence and counter-evidence."""
        support_score = 0
        oppose_score = 0
        analysis = []

        for e in evidence:
            # Simple overlap-based support check
            claim_words = set(claim.lower().split())
            evidence_words = set(e.lower().split())
            overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)
            support_score += overlap
            analysis.append({"type": "supporting", "evidence": e, "relevance": round(overlap, 3)})

        if counter_evidence:
            for ce in counter_evidence:
                claim_words = set(claim.lower().split())
                ce_words = set(ce.lower().split())
                overlap = len(claim_words & ce_words) / max(len(claim_words), 1)
                oppose_score += overlap
                analysis.append({"type": "opposing", "evidence": ce, "relevance": round(overlap, 3)})

        total = support_score + oppose_score + 1e-8
        verdict_score = support_score / total

        if verdict_score > 0.65:
            verdict = "SUPPORTED"
        elif verdict_score < 0.35:
            verdict = "REFUTED"
        else:
            verdict = "UNCERTAIN"

        return {
            "claim": claim,
            "verdict": verdict,
            "support_score": round(support_score, 3),
            "oppose_score": round(oppose_score, 3),
            "confidence": round(verdict_score, 3),
            "analysis": analysis,
        }

    def multi_step_deduction(self, premises: list[str], query: str) -> dict:
        """Multi-step logical deduction from premises."""
        # Simple rule-based deduction engine
        facts = {}
        rules = []

        for p in premises:
            p_lower = p.lower()
            if " is " in p_lower and " if " not in p_lower:
                parts = p_lower.split(" is ", 1)
                facts[parts[0].strip()] = parts[1].strip()
            elif "if " in p_lower and " then " in p_lower:
                condition = p_lower.split("if ")[1].split(" then ")[0].strip()
                conclusion = p_lower.split(" then ")[1].strip()
                rules.append({"condition": condition, "conclusion": conclusion})
            elif " are " in p_lower:
                parts = p_lower.split(" are ", 1)
                facts[parts[0].strip()] = parts[1].strip()

        # Apply rules
        deduction_steps = []
        for _ in range(5):  # Max 5 inference rounds
            new_facts = {}
            for rule in rules:
                # Check if condition is met
                for fact_key, fact_val in facts.items():
                    if rule["condition"] in fact_key or rule["condition"] in fact_val:
                        conclusion_parts = rule["conclusion"].split(" is ", 1) if " is " in rule["conclusion"] else [rule["conclusion"], "true"]
                        key = conclusion_parts[0].strip()
                        val = conclusion_parts[1].strip() if len(conclusion_parts) > 1 else "true"
                        if key not in facts:
                            new_facts[key] = val
                            deduction_steps.append({
                                "rule": f"if {rule['condition']} then {rule['conclusion']}",
                                "matched_fact": f"{fact_key} is {fact_val}",
                                "derived": f"{key} is {val}",
                            })
            if not new_facts:
                break
            facts.update(new_facts)

        # Check query
        query_lower = query.lower()
        answer = "Unknown"
        for key, val in facts.items():
            if key in query_lower or query_lower in key:
                answer = f"{key} is {val}"
                break

        return {
            "premises": premises,
            "query": query,
            "known_facts": facts,
            "deduction_steps": deduction_steps,
            "answer": answer,
            "total_facts_derived": len(deduction_steps),
        }


class ReasoningModule(BaseModule):
    name = "Chain-of-Thought & Reasoning"
    description = "CoT, Tree-of-Thought, self-verification, multi-step deduction"

    def run(self) -> dict:
        self.print_header()
        results = {}
        engine = ReasoningEngine()

        # --- 1. Chain of Thought ---
        with self.benchmark("Chain-of-Thought reasoning") as bm:
            cot = engine.chain_of_thought(
                problem="Should we deploy the new ML model to production?",
                steps=[
                    "Current model accuracy is 89%. New model shows 93% on test set.",
                    "New model latency is 45ms vs current 30ms — 50% slower.",
                    "A/B test on 5% traffic showed 12% improvement in user satisfaction.",
                    "Rollback plan exists: feature flag can revert in <1 minute.",
                    "Cost increase is ~$200/month for the larger model.",
                    "Conclusion: Deploy with monitoring. Accuracy gain justifies latency tradeoff.",
                ],
            )
            bm.metrics = {"steps": cot["total_steps"], "confidence": cot["final_confidence"]}
            results["chain_of_thought"] = cot

        print(f"\n  Chain-of-Thought: {cot['problem']}")
        for step in cot["chain"]:
            bar = "█" * int(step["confidence"] * 20)
            print(f"    Step {step['step']}: {step['thought'][:60]}...")
            print(f"           confidence: {bar} {step['cumulative_confidence']:.3f}")

        # --- 2. Tree of Thought ---
        with self.benchmark("Tree-of-Thought exploration") as bm:
            tot = engine.tree_of_thought(
                problem="Best architecture for a real-time recommendation system?",
                branches=[
                    [
                        "Approach A: Collaborative filtering with matrix factorization",
                        "Pre-compute user-item matrix, update hourly",
                        "Fast lookups but cold-start problem for new users",
                    ],
                    [
                        "Approach B: Deep learning with transformer-based model",
                        "Train on user interaction sequences",
                        "Better personalization but higher compute cost and latency",
                    ],
                    [
                        "Approach C: Hybrid — collaborative filtering + content-based + real-time features",
                        "Use CF for warm users, content-based for cold start",
                        "Add real-time session features via feature store",
                        "Most flexible but complex to maintain",
                    ],
                ],
            )
            bm.metrics = {"paths": tot["paths_explored"], "best_confidence": tot["best_confidence"]}
            results["tree_of_thought"] = tot

        print(f"\n  Tree-of-Thought: {tot['problem']}")
        for path in tot["all_paths"]:
            marker = " ← SELECTED" if path["branch"] == tot["selected_path"] else ""
            print(f"    Branch {path['branch']} (conf={path['confidence']:.3f}){marker}")
            for s in path["steps"]:
                print(f"      → {s['thought'][:65]}...")

        # --- 3. Self-Verification ---
        with self.benchmark("Self-verification check") as bm:
            verification = engine.self_verify(
                claim="Transformer models are more efficient than RNNs for long sequences",
                evidence=[
                    "Transformers process all positions in parallel using self-attention",
                    "Self-attention has O(n²) complexity but is highly parallelizable on GPUs",
                    "RNNs must process tokens sequentially, creating a bottleneck",
                    "BERT and GPT models handle sequences of 512-4096 tokens effectively",
                ],
                counter_evidence=[
                    "Self-attention has quadratic memory complexity O(n²)",
                    "For very long sequences (>10k tokens), RNN variants may use less memory",
                    "Linear attention approximations are needed for efficiency",
                ],
            )
            bm.metrics = {"verdict": verification["verdict"], "confidence": verification["confidence"]}
            results["self_verification"] = verification

        print(f"\n  Self-Verification:")
        print(f"    Claim: {verification['claim']}")
        print(f"    Verdict: {verification['verdict']} (confidence={verification['confidence']:.3f})")
        print(f"    Support: {verification['support_score']:.3f} | Oppose: {verification['oppose_score']:.3f}")

        # --- 4. Multi-step Deduction ---
        with self.benchmark("Multi-step logical deduction") as bm:
            deduction = engine.multi_step_deduction(
                premises=[
                    "Python is a programming language",
                    "PyTorch is a deep learning framework",
                    "If a tool is a deep learning framework then it is used for AI",
                    "If something is used for AI then it requires compute resources",
                    "Transformers are neural network architectures",
                    "If something is a neural network architecture then it is used for AI",
                ],
                query="Are transformers used for AI?",
            )
            bm.metrics = {"facts_derived": deduction["total_facts_derived"]}
            results["deduction"] = deduction

        print(f"\n  Multi-step Deduction:")
        print(f"    Query: {deduction['query']}")
        print(f"    Answer: {deduction['answer']}")
        print(f"    Facts derived: {deduction['total_facts_derived']}")
        for step in deduction["deduction_steps"]:
            print(f"      Rule: {step['rule']}")
            print(f"      → Derived: {step['derived']}")

        # --- 5. Reasoning trace summary ---
        with self.benchmark("Generate reasoning trace summary") as bm:
            total_thoughts = len(engine.thoughts)
            avg_confidence = sum(t.confidence for t in engine.thoughts.values()) / max(total_thoughts, 1)
            trace = {
                "total_thought_nodes": total_thoughts,
                "avg_confidence": round(avg_confidence, 3),
                "strategies_used": ["chain_of_thought", "tree_of_thought", "self_verification", "deduction"],
                "thought_tree_depth": max((len(self._get_ancestors(engine, t.id)) for t in engine.thoughts.values()), default=0),
            }
            bm.metrics = {"thought_nodes": total_thoughts}
            results["trace_summary"] = trace

        print(f"\n  Reasoning Trace: {total_thoughts} thought nodes, avg confidence={avg_confidence:.3f}")

        self.print_benchmark_table()
        return results

    def _get_ancestors(self, engine: ReasoningEngine, node_id: int) -> list[int]:
        ancestors = []
        current = node_id
        while current and current in engine.thoughts:
            parent = engine.thoughts[current].parent_id
            if parent:
                ancestors.append(parent)
            current = parent
        return ancestors
