"""
Module 7: Self-Evolution Engine
=================================
The crown jewel — implements self-reflection, performance tracking,
strategy mutation, and autonomous self-improvement loops.
"""

import time
import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from .base import BaseModule


@dataclass
class Strategy:
    """A mutable strategy the agent can evolve."""
    id: str
    name: str
    parameters: dict
    fitness: float = 0.0
    generation: int = 0
    parent_id: Optional[str] = None
    mutation_history: list[str] = field(default_factory=list)

    def mutate(self, mutation_rate: float = 0.3) -> "Strategy":
        """Create a mutated copy of this strategy."""
        new_params = dict(self.parameters)
        mutations = []
        for key, value in new_params.items():
            if random.random() < mutation_rate:
                if isinstance(value, float):
                    delta = random.gauss(0, value * 0.2)
                    new_params[key] = max(0.01, value + delta)
                    mutations.append(f"{key}: {value:.3f} → {new_params[key]:.3f}")
                elif isinstance(value, int):
                    delta = random.randint(-2, 2)
                    new_params[key] = max(1, value + delta)
                    mutations.append(f"{key}: {value} → {new_params[key]}")

        new_id = hashlib.md5(str(new_params).encode()).hexdigest()[:8]
        return Strategy(
            id=new_id, name=f"{self.name}_v{self.generation+1}",
            parameters=new_params, generation=self.generation + 1,
            parent_id=self.id, mutation_history=mutations,
        )


@dataclass
class PerformanceLog:
    task: str
    strategy_id: str
    score: float
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class SelfReflector:
    """Analyzes agent performance and identifies improvement areas."""

    def reflect(self, logs: list[PerformanceLog]) -> dict:
        if not logs:
            return {"status": "no data", "recommendations": []}

        avg_score = sum(l.score for l in logs) / len(logs)
        avg_latency = sum(l.latency_ms for l in logs) / len(logs)
        score_trend = self._compute_trend([l.score for l in logs])
        latency_trend = self._compute_trend([l.latency_ms for l in logs])

        # Identify weak areas
        weaknesses = []
        strengths = []
        for log in logs:
            if log.score < avg_score * 0.8:
                weaknesses.append(log.task)
            elif log.score > avg_score * 1.1:
                strengths.append(log.task)

        recommendations = []
        if avg_score < 0.6:
            recommendations.append("Strategy needs significant improvement — consider larger mutations")
        if score_trend < -0.05:
            recommendations.append("Performance declining — revert to previous best strategy")
        if latency_trend > 0.1:
            recommendations.append("Latency increasing — optimize tool usage or reduce steps")
        if len(weaknesses) > len(logs) * 0.3:
            recommendations.append(f"Weak on {len(weaknesses)} task types — add specialized strategies")
        if avg_score > 0.8 and score_trend > 0:
            recommendations.append("Strong performance — reduce mutation rate for stability")

        return {
            "avg_score": round(avg_score, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "score_trend": round(score_trend, 4),
            "latency_trend": round(latency_trend, 4),
            "strengths": strengths[:3],
            "weaknesses": weaknesses[:3],
            "recommendations": recommendations,
            "total_evaluations": len(logs),
        }

    def _compute_trend(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        return numerator / (denominator + 1e-8)


class EvolutionEngine:
    """Manages strategy evolution through mutation, evaluation, and selection."""

    def __init__(self, population_size: int = 6):
        self.population_size = population_size
        self.population: list[Strategy] = []
        self.performance_logs: list[PerformanceLog] = []
        self.reflector = SelfReflector()
        self.generation = 0
        self.best_ever: Optional[Strategy] = None

    def initialize_population(self, base_strategy: Strategy):
        """Create initial population through mutations of base strategy."""
        self.population = [base_strategy]
        for _ in range(self.population_size - 1):
            mutant = base_strategy.mutate(mutation_rate=0.5)
            self.population.append(mutant)

    def evaluate(self, strategy: Strategy, tasks: list[dict]) -> float:
        """Evaluate a strategy on a set of tasks."""
        scores = []
        for task in tasks:
            start = time.perf_counter()
            score = self._simulate_task_performance(strategy, task)
            elapsed = (time.perf_counter() - start) * 1000

            self.performance_logs.append(PerformanceLog(
                task=task["name"], strategy_id=strategy.id,
                score=score, latency_ms=elapsed,
            ))
            scores.append(score)

        fitness = sum(scores) / len(scores) if scores else 0
        strategy.fitness = fitness
        return fitness

    def _simulate_task_performance(self, strategy: Strategy, task: dict) -> float:
        """Simulate how well a strategy performs on a task."""
        base_score = task.get("difficulty", 0.5)
        params = strategy.parameters

        # Score based on parameter-task alignment
        score = base_score
        if "temperature" in params:
            # Lower temperature = more deterministic = better for factual tasks
            if task.get("type") == "factual":
                score *= (1.0 - params["temperature"] * 0.3)
            else:
                score *= (0.7 + params["temperature"] * 0.3)

        if "retrieval_depth" in params:
            score *= min(1.0, 0.5 + params["retrieval_depth"] * 0.1)

        if "reasoning_steps" in params:
            # More steps help complex tasks but hurt simple ones
            complexity = task.get("complexity", 0.5)
            optimal_steps = max(1, int(complexity * 10))
            step_diff = abs(params["reasoning_steps"] - optimal_steps)
            score *= max(0.3, 1.0 - step_diff * 0.1)

        # Add noise
        score += random.gauss(0, 0.05)
        return max(0.0, min(1.0, score))

    def evolve(self, tasks: list[dict]) -> dict:
        """Run one generation of evolution."""
        self.generation += 1

        # Evaluate all strategies
        for strategy in self.population:
            self.evaluate(strategy, tasks)

        # Sort by fitness
        self.population.sort(key=lambda s: s.fitness, reverse=True)

        # Track best ever
        if not self.best_ever or self.population[0].fitness > self.best_ever.fitness:
            self.best_ever = self.population[0]

        # Selection: keep top half
        survivors = self.population[:self.population_size // 2]

        # Produce offspring through mutation
        offspring = []
        for parent in survivors:
            child = parent.mutate(mutation_rate=0.3)
            offspring.append(child)

        # New population = survivors + offspring (+ best ever for elitism)
        self.population = survivors + offspring
        if self.best_ever not in self.population:
            self.population[-1] = Strategy(
                id=self.best_ever.id, name=self.best_ever.name,
                parameters=dict(self.best_ever.parameters),
                fitness=self.best_ever.fitness, generation=self.best_ever.generation,
            )

        # Self-reflect
        reflection = self.reflector.reflect(self.performance_logs[-len(tasks) * len(self.population):])

        return {
            "generation": self.generation,
            "best_fitness": round(self.population[0].fitness, 4),
            "avg_fitness": round(sum(s.fitness for s in self.population) / len(self.population), 4),
            "worst_fitness": round(self.population[-1].fitness, 4),
            "best_strategy": {"id": self.population[0].id, "params": self.population[0].parameters},
            "best_ever_fitness": round(self.best_ever.fitness, 4),
            "reflection": reflection,
            "population_size": len(self.population),
        }


class SelfEvolutionModule(BaseModule):
    name = "Self-Evolution Engine"
    description = "Strategy mutation, fitness evaluation, selection, self-reflection, improvement loops"

    TASKS = [
        {"name": "factual_qa", "type": "factual", "difficulty": 0.7, "complexity": 0.3},
        {"name": "creative_writing", "type": "creative", "difficulty": 0.6, "complexity": 0.7},
        {"name": "code_generation", "type": "technical", "difficulty": 0.8, "complexity": 0.8},
        {"name": "summarization", "type": "factual", "difficulty": 0.5, "complexity": 0.4},
        {"name": "multi_step_reasoning", "type": "reasoning", "difficulty": 0.9, "complexity": 0.9},
        {"name": "data_analysis", "type": "technical", "difficulty": 0.7, "complexity": 0.6},
    ]

    def run(self) -> dict:
        self.print_header()
        results = {}
        random.seed(42)

        # --- 1. Initialize Base Strategy ---
        with self.benchmark("Initialize base strategy & population") as bm:
            base = Strategy(
                id="base_001", name="default_strategy",
                parameters={
                    "temperature": 0.7,
                    "retrieval_depth": 3,
                    "reasoning_steps": 5,
                    "confidence_threshold": 0.6,
                    "max_retries": 2,
                },
                generation=0,
            )
            engine = EvolutionEngine(population_size=6)
            engine.initialize_population(base)
            bm.metrics = {"population": len(engine.population)}
            results["initial_population"] = [
                {"id": s.id, "name": s.name, "params": s.parameters}
                for s in engine.population
            ]

        print(f"\n  Initial Population ({len(engine.population)} strategies):")
        for s in engine.population:
            print(f"    {s.id}: temp={s.parameters.get('temperature', 0):.2f}, "
                  f"depth={s.parameters.get('retrieval_depth', 0)}, "
                  f"steps={s.parameters.get('reasoning_steps', 0)}")

        # --- 2. Run Evolution (10 generations) ---
        evolution_history = []
        num_generations = 10

        for gen in range(num_generations):
            with self.benchmark(f"Generation {gen + 1}") as bm:
                gen_result = engine.evolve(self.TASKS)
                evolution_history.append(gen_result)
                bm.metrics = {
                    "best_fitness": gen_result["best_fitness"],
                    "avg_fitness": gen_result["avg_fitness"],
                }

        results["evolution_history"] = evolution_history

        print(f"\n  Evolution Progress ({num_generations} generations):")
        print(f"  {'Gen':>4} {'Best':>8} {'Avg':>8} {'Worst':>8} {'Best Ever':>10}")
        print(f"  {'─'*42}")
        for h in evolution_history:
            bar = "█" * int(h["best_fitness"] * 20)
            print(f"  {h['generation']:>4} {h['best_fitness']:>8.4f} {h['avg_fitness']:>8.4f} "
                  f"{h['worst_fitness']:>8.4f} {h['best_ever_fitness']:>10.4f}  {bar}")

        # --- 3. Self-Reflection Analysis ---
        with self.benchmark("Self-reflection analysis") as bm:
            final_reflection = engine.reflector.reflect(engine.performance_logs)
            bm.metrics = {"evaluations": final_reflection["total_evaluations"]}
            results["final_reflection"] = final_reflection

        print(f"\n  Self-Reflection Report:")
        print(f"    Avg Score: {final_reflection['avg_score']:.4f}")
        print(f"    Score Trend: {final_reflection['score_trend']:+.4f}")
        print(f"    Latency Trend: {final_reflection['latency_trend']:+.4f}")
        print(f"    Strengths: {final_reflection['strengths']}")
        print(f"    Weaknesses: {final_reflection['weaknesses']}")
        print(f"    Recommendations:")
        for r in final_reflection["recommendations"]:
            print(f"      → {r}")

        # --- 4. Best Strategy Analysis ---
        with self.benchmark("Best strategy analysis") as bm:
            best = engine.best_ever
            improvement = evolution_history[-1]["best_fitness"] - evolution_history[0]["best_fitness"]
            best_info = {
                "id": best.id,
                "name": best.name,
                "parameters": best.parameters,
                "fitness": round(best.fitness, 4),
                "generation_found": best.generation,
                "improvement_over_base": round(improvement, 4),
                "mutation_history": best.mutation_history[:5],
            }
            bm.metrics = {"best_fitness": best.fitness, "improvement": round(improvement, 4)}
            results["best_strategy"] = best_info

        print(f"\n  Best Strategy Found:")
        print(f"    ID: {best.id}")
        print(f"    Fitness: {best.fitness:.4f}")
        print(f"    Generation: {best.generation}")
        print(f"    Improvement: {improvement:+.4f}")
        print(f"    Parameters: {json.dumps(best.parameters, indent=6)}" if hasattr(best, 'parameters') else "")

        # --- 5. Evolution convergence analysis ---
        with self.benchmark("Convergence analysis") as bm:
            fitness_values = [h["best_fitness"] for h in evolution_history]
            converged = False
            convergence_gen = num_generations
            for i in range(2, len(fitness_values)):
                if abs(fitness_values[i] - fitness_values[i-1]) < 0.001 and abs(fitness_values[i] - fitness_values[i-2]) < 0.001:
                    converged = True
                    convergence_gen = i + 1
                    break

            convergence = {
                "converged": converged,
                "convergence_generation": convergence_gen if converged else None,
                "final_best": round(fitness_values[-1], 4),
                "initial_best": round(fitness_values[0], 4),
                "total_improvement": round(fitness_values[-1] - fitness_values[0], 4),
                "improvement_pct": round((fitness_values[-1] - fitness_values[0]) / max(fitness_values[0], 0.01) * 100, 2),
            }
            bm.metrics = {"converged": converged, "improvement_pct": convergence["improvement_pct"]}
            results["convergence"] = convergence

        print(f"\n  Convergence: {'Yes' if converged else 'Not yet'} "
              f"(improvement: {convergence['improvement_pct']:+.2f}%)")

        self.print_benchmark_table()
        return results


# Need json import for the best strategy display
import json
