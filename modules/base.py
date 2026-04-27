"""Base module class for all GenAI concept demonstrations."""

import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    """Captures timing and quality metrics for a single operation."""
    operation: str
    duration_ms: float
    metrics: dict = field(default_factory=dict)
    output_preview: str = ""

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2),
            "metrics": self.metrics,
            "output_preview": self.output_preview[:200],
        }


class BaseModule(ABC):
    """Every GenAI concept module inherits from this."""

    name: str = "base"
    description: str = ""

    def __init__(self):
        self.benchmarks: list[BenchmarkResult] = []

    def benchmark(self, operation: str):
        """Context manager to time an operation."""
        return _BenchmarkContext(self, operation)

    def add_benchmark(self, result: BenchmarkResult):
        self.benchmarks.append(result)

    @abstractmethod
    def run(self) -> dict:
        """Execute the module demo and return results dict."""
        ...

    def get_report(self) -> dict:
        return {
            "module": self.name,
            "description": self.description,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "total_time_ms": round(sum(b.duration_ms for b in self.benchmarks), 2),
        }

    def print_header(self):
        width = 70
        print(f"\n{'='*width}")
        print(f"  {self.name.upper()}")
        print(f"  {self.description}")
        print(f"{'='*width}")

    def print_benchmark_table(self):
        if not self.benchmarks:
            return
        print(f"\n{'─'*60}")
        print(f"  {'Operation':<35} {'Time (ms)':>10}  {'Key Metric'}")
        print(f"{'─'*60}")
        for b in self.benchmarks:
            metric_str = ""
            if b.metrics:
                key = list(b.metrics.keys())[0]
                val = b.metrics[key]
                metric_str = f"{key}={val}"
            print(f"  {b.operation:<35} {b.duration_ms:>10.2f}  {metric_str}")
        total = sum(b.duration_ms for b in self.benchmarks)
        print(f"{'─'*60}")
        print(f"  {'TOTAL':<35} {total:>10.2f}")
        print()


class _BenchmarkContext:
    def __init__(self, module: BaseModule, operation: str):
        self.module = module
        self.operation = operation
        self.result = BenchmarkResult(operation=operation, duration_ms=0)

    def __enter__(self):
        self.start = time.perf_counter()
        return self.result

    def __exit__(self, *args):
        self.result.duration_ms = (time.perf_counter() - self.start) * 1000
        self.module.add_benchmark(self.result)
