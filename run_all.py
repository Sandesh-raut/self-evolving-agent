#!/usr/bin/env python3
"""
Self-Evolving AI Agent — Master Runner
========================================
Executes all GenAI concept modules with benchmarking and generates
a comprehensive performance report.

Usage:
    python run_all.py              # Run all modules
    python run_all.py --module 1   # Run specific module (1-11)
    python run_all.py --fast       # Skip HuggingFace model-dependent modules
"""

import sys
import os
import json
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_modules(fast_mode: bool = False):
    """Import and return all available modules."""
    from modules.tokenization import TokenizationModule
    from modules.embeddings import EmbeddingsModule
    from modules.rag_pipeline import RAGModule
    from modules.prompt_engineering import PromptEngineeringModule
    from modules.reasoning import ReasoningModule
    from modules.agent_architecture import AgentArchitectureModule
    from modules.self_evolution import SelfEvolutionModule
    from modules.multimodal import MultiModalModule
    from modules.finetuning import FineTuningModule
    from modules.evaluation import EvaluationModule
    from modules.agent_swarm import AgentSwarmModule

    all_modules = [
        (1, "Tokenization", TokenizationModule),
        (2, "Embeddings", EmbeddingsModule),
        (3, "RAG Pipeline", RAGModule),
        (4, "Prompt Engineering", PromptEngineeringModule),
        (5, "Reasoning", ReasoningModule),
        (6, "Agent Architecture", AgentArchitectureModule),
        (7, "Self-Evolution", SelfEvolutionModule),
        (8, "Multi-Modal", MultiModalModule),
        (9, "Fine-Tuning", FineTuningModule),
        (10, "Evaluation", EvaluationModule),
        (11, "Agent Swarm", AgentSwarmModule),
    ]

    if fast_mode:
        # Skip modules that require HuggingFace models (they still work via fallback,
        # but --fast means "no large downloads")
        pass  # All modules have fallbacks, so all can run

    return all_modules


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            🧠  SELF-EVOLVING AI AGENT  🧠                            ║
║            ─────────────────────────────                             ║
║            A Comprehensive GenAI Concepts Showcase                   ║
║                                                                      ║
║            Covering 11 Core AI/ML Concepts:                          ║
║            Tokenization · Embeddings · RAG · Prompt Engineering      ║
║            Reasoning · Agent Architecture · Self-Evolution           ║
║            Multi-Modal · Fine-Tuning · Evaluation · Agent Swarm      ║
║                                                                      ║
║            Built by Sandesh Raut                                     ║
║            github.com/sandesh37                                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_module(module_class, module_name: str) -> dict:
    """Run a single module and return its results."""
    try:
        instance = module_class()
        result = instance.run()
        report = instance.get_report()
        return {"status": "success", "result": result, "report": report}
    except Exception as e:
        print(f"\n  [ERROR] {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def generate_summary_report(all_reports: list[dict]) -> dict:
    """Generate a comprehensive summary report."""
    total_time = 0
    total_benchmarks = 0
    module_summaries = []

    for report in all_reports:
        if report.get("status") != "success":
            continue
        r = report["report"]
        total_time += r.get("total_time_ms", 0)
        num_bm = len(r.get("benchmarks", []))
        total_benchmarks += num_bm
        module_summaries.append({
            "module": r["module"],
            "time_ms": r["total_time_ms"],
            "benchmarks": num_bm,
            "description": r["description"],
        })

    return {
        "total_modules_run": len(module_summaries),
        "total_time_ms": round(total_time, 2),
        "total_time_sec": round(total_time / 1000, 2),
        "total_benchmarks": total_benchmarks,
        "module_summaries": module_summaries,
    }


def print_final_report(summary: dict, output_dir: str):
    """Print the final summary report."""
    width = 70
    print(f"\n{'═'*width}")
    print(f"  FINAL PERFORMANCE REPORT")
    print(f"{'═'*width}")
    print(f"  Modules Run:       {summary['total_modules_run']}")
    print(f"  Total Benchmarks:  {summary['total_benchmarks']}")
    print(f"  Total Time:        {summary['total_time_sec']:.2f}s ({summary['total_time_ms']:.0f}ms)")
    print(f"\n  {'Module':<30} {'Time (ms)':>12} {'Benchmarks':>12}")
    print(f"  {'─'*55}")
    for m in summary["module_summaries"]:
        pct = (m["time_ms"] / max(summary["total_time_ms"], 1)) * 100
        bar = "█" * max(1, int(pct / 3))
        print(f"  {m['module']:<30} {m['time_ms']:>12.2f} {m['benchmarks']:>12}  {bar}")
    print(f"  {'─'*55}")
    print(f"  {'TOTAL':<30} {summary['total_time_ms']:>12.2f} {summary['total_benchmarks']:>12}")
    print(f"\n  Report saved to: {output_dir}/benchmark_report.json")
    print(f"{'═'*width}\n")


def main():
    parser = argparse.ArgumentParser(description="Self-Evolving AI Agent — Run all GenAI modules")
    parser.add_argument("--module", "-m", type=int, help="Run specific module (1-11)")
    parser.add_argument("--fast", action="store_true", help="Skip heavy model downloads")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    args = parser.parse_args()

    print_banner()

    # Setup output dir
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    os.makedirs(output_dir, exist_ok=True)

    modules = get_modules(fast_mode=args.fast)

    if args.module:
        modules = [(num, name, cls) for num, name, cls in modules if num == args.module]
        if not modules:
            print(f"  [ERROR] Module {args.module} not found. Valid: 1-11")
            sys.exit(1)

    all_reports = []
    start_time = time.perf_counter()

    for num, name, cls in modules:
        print(f"\n{'▶'*3} Running Module {num}/10: {name}")
        report = run_module(cls, name)
        all_reports.append(report)

    total_elapsed = (time.perf_counter() - start_time) * 1000

    # Generate and print summary
    summary = generate_summary_report(all_reports)
    summary["wall_clock_ms"] = round(total_elapsed, 2)
    print_final_report(summary, output_dir)

    # Save report
    report_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_path, "w") as f:
        # Only save serializable data
        serializable = {
            "summary": summary,
            "module_reports": [
                r["report"] for r in all_reports if r.get("status") == "success"
            ],
        }
        json.dump(serializable, f, indent=2, default=str)

    print(f"  All done! {summary['total_modules_run']} modules, {summary['total_benchmarks']} benchmarks.\n")


if __name__ == "__main__":
    main()
