"""
Module 10: Evaluation & Benchmarking Framework
=================================================
Comprehensive evaluation framework with metrics, benchmark suites,
statistical analysis, and automated performance reporting.
"""

import math
import time
import random
import statistics
from collections import defaultdict
from .base import BaseModule


class MetricsCalculator:
    """Compute standard ML evaluation metrics."""

    @staticmethod
    def accuracy(y_true: list[int], y_pred: list[int]) -> float:
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true) if y_true else 0

    @staticmethod
    def precision_recall_f1(y_true: list[int], y_pred: list[int], num_classes: int) -> dict:
        results = {}
        for c in range(num_classes):
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            results[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        return results

    @staticmethod
    def bleu_score(reference: list[str], hypothesis: list[str], n: int = 4) -> float:
        """Simplified BLEU score computation."""
        from collections import Counter
        scores = []
        for i in range(1, n + 1):
            ref_ngrams = Counter(zip(*[reference[j:] for j in range(i)]))
            hyp_ngrams = Counter(zip(*[hypothesis[j:] for j in range(i)]))
            overlap = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())
            scores.append(overlap / total if total > 0 else 0)

        if not scores or min(scores) == 0:
            return 0.0

        # Geometric mean
        log_avg = sum(math.log(max(s, 1e-10)) for s in scores) / len(scores)
        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(reference) / max(len(hypothesis), 1)))
        return round(bp * math.exp(log_avg), 4)

    @staticmethod
    def rouge_l(reference: str, hypothesis: str) -> float:
        """ROUGE-L using longest common subsequence."""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        m, n = len(ref_words), len(hyp_words)
        if m == 0 or n == 0:
            return 0.0

        # LCS length
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs = dp[m][n]
        prec = lcs / n
        rec = lcs / m
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return round(f1, 4)

    @staticmethod
    def perplexity(log_probs: list[float]) -> float:
        """Compute perplexity from log probabilities."""
        avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else 0
        return round(math.exp(-avg_log_prob), 4)


class BenchmarkSuite:
    """Standardized benchmark suite for evaluating AI systems."""

    def __init__(self):
        self.results: dict[str, list[dict]] = defaultdict(list)

    def run_latency_benchmark(self, fn, name: str, iterations: int = 50) -> dict:
        """Measure function latency over multiple iterations."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        result = {
            "name": name,
            "iterations": iterations,
            "mean_ms": round(statistics.mean(times), 3),
            "median_ms": round(statistics.median(times), 3),
            "std_ms": round(statistics.stdev(times), 3) if len(times) > 1 else 0,
            "min_ms": round(min(times), 3),
            "max_ms": round(max(times), 3),
            "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 3),
            "p99_ms": round(sorted(times)[int(len(times) * 0.99)], 3),
        }
        self.results[name].append(result)
        return result

    def run_throughput_benchmark(self, fn, name: str, duration_sec: float = 2.0) -> dict:
        """Measure throughput (operations per second)."""
        count = 0
        start = time.perf_counter()
        while time.perf_counter() - start < duration_sec:
            fn()
            count += 1
        elapsed = time.perf_counter() - start
        return {
            "name": name,
            "operations": count,
            "duration_sec": round(elapsed, 3),
            "ops_per_sec": round(count / elapsed, 2),
        }

    def compare(self, results_a: dict, results_b: dict) -> dict:
        """Compare two benchmark results."""
        speedup = results_b["mean_ms"] / max(results_a["mean_ms"], 0.001)
        return {
            "comparison": f"{results_a['name']} vs {results_b['name']}",
            "speedup": round(speedup, 2),
            "a_mean_ms": results_a["mean_ms"],
            "b_mean_ms": results_b["mean_ms"],
            "winner": results_a["name"] if results_a["mean_ms"] < results_b["mean_ms"] else results_b["name"],
        }


class StatisticalAnalyzer:
    """Statistical analysis for benchmark results."""

    @staticmethod
    def confidence_interval(values: list[float], confidence: float = 0.95) -> tuple:
        n = len(values)
        if n < 2:
            return (0, 0)
        mean = statistics.mean(values)
        se = statistics.stdev(values) / math.sqrt(n)
        # Approximate z-score for 95% CI
        z = 1.96 if confidence == 0.95 else 2.576
        return (round(mean - z * se, 4), round(mean + z * se, 4))

    @staticmethod
    def effect_size(group_a: list[float], group_b: list[float]) -> float:
        """Cohen's d effect size."""
        n_a, n_b = len(group_a), len(group_b)
        if n_a < 2 or n_b < 2:
            return 0
        mean_a, mean_b = statistics.mean(group_a), statistics.mean(group_b)
        var_a, var_b = statistics.variance(group_a), statistics.variance(group_b)
        pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        if pooled_std == 0:
            return 0
        return round((mean_a - mean_b) / pooled_std, 4)


class EvaluationModule(BaseModule):
    name = "Evaluation & Benchmarking"
    description = "ML metrics, latency/throughput benchmarks, statistical analysis, comparison tables"

    def run(self) -> dict:
        self.print_header()
        results = {}
        random.seed(42)
        calc = MetricsCalculator()
        suite = BenchmarkSuite()
        stats = StatisticalAnalyzer()

        # --- 1. Classification Metrics ---
        with self.benchmark("Classification metrics computation") as bm:
            y_true = [random.randint(0, 2) for _ in range(200)]
            y_pred_good = [t if random.random() > 0.15 else random.randint(0, 2) for t in y_true]
            y_pred_bad = [t if random.random() > 0.5 else random.randint(0, 2) for t in y_true]

            acc_good = calc.accuracy(y_true, y_pred_good)
            acc_bad = calc.accuracy(y_true, y_pred_bad)
            prf_good = calc.precision_recall_f1(y_true, y_pred_good, 3)
            prf_bad = calc.precision_recall_f1(y_true, y_pred_bad, 3)

            classification = {
                "good_model": {"accuracy": round(acc_good, 4), "per_class": prf_good},
                "bad_model": {"accuracy": round(acc_bad, 4), "per_class": prf_bad},
            }
            bm.metrics = {"good_acc": round(acc_good, 4), "bad_acc": round(acc_bad, 4)}
            results["classification_metrics"] = classification

        print(f"\n  Classification Metrics (200 samples, 3 classes):")
        print(f"    Good Model: accuracy={acc_good:.4f}")
        print(f"    Bad Model:  accuracy={acc_bad:.4f}")
        print(f"    {'Model':>12} {'Class':>6} {'Prec':>7} {'Recall':>7} {'F1':>7}")
        print(f"    {'─'*45}")
        for c in range(3):
            print(f"    {'Good':>12} {c:>6} {prf_good[c]['precision']:>7.4f} {prf_good[c]['recall']:>7.4f} {prf_good[c]['f1']:>7.4f}")
        for c in range(3):
            print(f"    {'Bad':>12} {c:>6} {prf_bad[c]['precision']:>7.4f} {prf_bad[c]['recall']:>7.4f} {prf_bad[c]['f1']:>7.4f}")

        # --- 2. Text Generation Metrics ---
        with self.benchmark("Text generation metrics (BLEU, ROUGE-L)") as bm:
            test_pairs = [
                ("The transformer model uses self attention to process sequences efficiently",
                 "Transformer models use self attention for efficient sequence processing"),
                ("Machine learning algorithms learn patterns from data",
                 "ML algorithms discover patterns in training data"),
                ("Deep neural networks have multiple hidden layers",
                 "Deep learning networks contain several hidden layers"),
            ]
            gen_metrics = []
            for ref, hyp in test_pairs:
                bleu = calc.bleu_score(ref.split(), hyp.split())
                rouge = calc.rouge_l(ref, hyp)
                gen_metrics.append({"reference": ref[:50], "hypothesis": hyp[:50], "bleu": bleu, "rouge_l": rouge})

            bm.metrics = {"avg_bleu": round(sum(g["bleu"] for g in gen_metrics) / len(gen_metrics), 4)}
            results["generation_metrics"] = gen_metrics

        print(f"\n  Text Generation Metrics:")
        print(f"  {'Reference':<40} {'BLEU':>7} {'ROUGE-L':>8}")
        print(f"  {'─'*58}")
        for g in gen_metrics:
            print(f"  {g['reference']:<40} {g['bleu']:>7.4f} {g['rouge_l']:>8.4f}")

        # --- 3. Latency Benchmarks ---
        def task_fast():
            _ = sum(range(100))

        def task_medium():
            _ = [math.sqrt(i) for i in range(500)]

        def task_slow():
            _ = sorted([random.random() for _ in range(1000)])

        with self.benchmark("Latency benchmark suite") as bm:
            latency_fast = suite.run_latency_benchmark(task_fast, "fast_computation", 100)
            latency_medium = suite.run_latency_benchmark(task_medium, "medium_computation", 100)
            latency_slow = suite.run_latency_benchmark(task_slow, "slow_computation", 50)
            bm.metrics = {"benchmarks_run": 3}
            results["latency_benchmarks"] = [latency_fast, latency_medium, latency_slow]

        print(f"\n  Latency Benchmarks:")
        print(f"  {'Benchmark':<22} {'Mean':>8} {'Median':>8} {'P95':>8} {'P99':>8} {'Std':>8}")
        print(f"  {'─'*62}")
        for b in [latency_fast, latency_medium, latency_slow]:
            print(f"  {b['name']:<22} {b['mean_ms']:>8.3f} {b['median_ms']:>8.3f} "
                  f"{b['p95_ms']:>8.3f} {b['p99_ms']:>8.3f} {b['std_ms']:>8.3f}")

        # --- 4. Throughput Benchmarks ---
        with self.benchmark("Throughput benchmarks") as bm:
            tp_results = []
            for fn, name in [(task_fast, "fast"), (task_medium, "medium"), (task_slow, "slow")]:
                tp = suite.run_throughput_benchmark(fn, name, duration_sec=1.0)
                tp_results.append(tp)
            bm.metrics = {"benchmarks": len(tp_results)}
            results["throughput_benchmarks"] = tp_results

        print(f"\n  Throughput Benchmarks:")
        print(f"  {'Task':<15} {'Ops':>8} {'Ops/sec':>12}")
        print(f"  {'─'*38}")
        for t in tp_results:
            print(f"  {t['name']:<15} {t['operations']:>8} {t['ops_per_sec']:>12.2f}")

        # --- 5. Statistical Analysis ---
        with self.benchmark("Statistical analysis") as bm:
            group_a = [random.gauss(100, 10) for _ in range(50)]
            group_b = [random.gauss(105, 12) for _ in range(50)]

            ci_a = stats.confidence_interval(group_a)
            ci_b = stats.confidence_interval(group_b)
            effect = stats.effect_size(group_a, group_b)

            stat_analysis = {
                "group_a": {"mean": round(statistics.mean(group_a), 2), "std": round(statistics.stdev(group_a), 2), "ci_95": ci_a},
                "group_b": {"mean": round(statistics.mean(group_b), 2), "std": round(statistics.stdev(group_b), 2), "ci_95": ci_b},
                "effect_size_cohens_d": effect,
                "effect_interpretation": "small" if abs(effect) < 0.5 else "medium" if abs(effect) < 0.8 else "large",
            }
            bm.metrics = {"cohens_d": effect}
            results["statistical_analysis"] = stat_analysis

        print(f"\n  Statistical Analysis (A/B Test):")
        print(f"    Group A: mean={stat_analysis['group_a']['mean']}, std={stat_analysis['group_a']['std']}, CI={ci_a}")
        print(f"    Group B: mean={stat_analysis['group_b']['mean']}, std={stat_analysis['group_b']['std']}, CI={ci_b}")
        print(f"    Cohen's d: {effect} ({stat_analysis['effect_interpretation']} effect)")

        # --- 6. Perplexity Demo ---
        with self.benchmark("Perplexity computation") as bm:
            log_probs_good = [math.log(random.uniform(0.3, 0.9)) for _ in range(50)]
            log_probs_bad = [math.log(random.uniform(0.05, 0.3)) for _ in range(50)]
            ppl_good = calc.perplexity(log_probs_good)
            ppl_bad = calc.perplexity(log_probs_bad)
            bm.metrics = {"good_ppl": ppl_good, "bad_ppl": ppl_bad}
            results["perplexity"] = {"good_model": ppl_good, "bad_model": ppl_bad}

        print(f"\n  Perplexity (lower = better):")
        print(f"    Good model: {ppl_good:.4f}")
        print(f"    Bad model:  {ppl_bad:.4f}")

        self.print_benchmark_table()
        return results
