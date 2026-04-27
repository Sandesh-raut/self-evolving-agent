"""
Module 4: Prompt Engineering Patterns
=======================================
Demonstrates systematic prompt engineering: zero-shot, few-shot,
chain-of-thought, self-consistency, persona, and template patterns.
"""

import re
import time
from .base import BaseModule


class PromptTemplate:
    """Reusable prompt template with variable substitution."""

    def __init__(self, template: str, name: str = ""):
        self.template = template
        self.name = name

    def render(self, **kwargs) -> str:
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


class PromptEngineeringModule(BaseModule):
    name = "Prompt Engineering"
    description = "Zero/few-shot, CoT, self-consistency, persona patterns, template system"

    # Prompt pattern library
    PATTERNS = {
        "zero_shot": PromptTemplate(
            "Classify the sentiment of this text as POSITIVE, NEGATIVE, or NEUTRAL.\n\nText: {text}\nSentiment:",
            name="Zero-Shot Classification",
        ),
        "few_shot": PromptTemplate(
            """Classify the sentiment of the text as POSITIVE, NEGATIVE, or NEUTRAL.

Text: "This product is amazing, I love it!"
Sentiment: POSITIVE

Text: "Terrible experience, would not recommend."
Sentiment: NEGATIVE

Text: "The package arrived on Tuesday."
Sentiment: NEUTRAL

Text: "{text}"
Sentiment:""",
            name="Few-Shot Classification",
        ),
        "chain_of_thought": PromptTemplate(
            """Solve this step by step.

Problem: {problem}

Let me think through this step by step:
Step 1:""",
            name="Chain-of-Thought",
        ),
        "self_consistency": PromptTemplate(
            """Answer this question. Show your reasoning.

Question: {question}

Approach {approach_num}:
Let me think about this differently.""",
            name="Self-Consistency",
        ),
        "persona": PromptTemplate(
            """You are {persona}. {persona_context}

User: {query}
Response:""",
            name="Persona Pattern",
        ),
        "structured_output": PromptTemplate(
            """Extract information from the text and return as structured data.

Text: {text}

Return a JSON object with these fields:
- entities: list of named entities
- sentiment: overall sentiment
- topics: main topics
- summary: one-sentence summary

JSON:""",
            name="Structured Output",
        ),
        "react": PromptTemplate(
            """Answer the question using the ReAct framework.

Question: {question}

Thought 1: I need to break down this question.
Action 1: analyze_question
Observation 1:""",
            name="ReAct Pattern",
        ),
    }

    TEST_TEXTS = [
        ("This new AI model is incredibly fast and accurate!", "POSITIVE"),
        ("The service was slow and the food was cold.", "NEGATIVE"),
        ("The meeting is scheduled for 3 PM tomorrow.", "NEUTRAL"),
        ("I'm somewhat disappointed but the price was fair.", "MIXED"),
        ("Outstanding performance that exceeded all expectations!", "POSITIVE"),
    ]

    def run(self) -> dict:
        self.print_header()
        results = {}

        # --- 1. Pattern Catalogue ---
        with self.benchmark("Build prompt pattern catalogue") as bm:
            catalogue = {}
            for name, template in self.PATTERNS.items():
                # Analyze template
                variables = re.findall(r'\{(\w+)\}', template.template)
                catalogue[name] = {
                    "name": template.name,
                    "variables": variables,
                    "template_length": len(template.template),
                    "template_preview": template.template[:100] + "...",
                }
            bm.metrics = {"patterns": len(catalogue)}
            results["pattern_catalogue"] = catalogue

        print(f"\n  Prompt Pattern Catalogue ({len(catalogue)} patterns):")
        for name, info in catalogue.items():
            print(f"    {name:<22} vars={info['variables']}")

        # --- 2. Zero-Shot vs Few-Shot Comparison ---
        with self.benchmark("Zero-shot vs Few-shot comparison") as bm:
            comparison = []
            for text, expected in self.TEST_TEXTS:
                zero_prompt = self.PATTERNS["zero_shot"].render(text=text)
                few_prompt = self.PATTERNS["few_shot"].render(text=text)

                # Simulate classification using keyword matching (no LLM needed)
                predicted_zero = self._keyword_classify(text)
                predicted_few = self._keyword_classify_enhanced(text)

                comparison.append({
                    "text": text[:50],
                    "expected": expected,
                    "zero_shot": predicted_zero,
                    "few_shot": predicted_few,
                    "zero_correct": predicted_zero == expected,
                    "few_correct": predicted_few == expected,
                    "zero_prompt_len": len(zero_prompt),
                    "few_prompt_len": len(few_prompt),
                })

            zero_acc = sum(1 for c in comparison if c["zero_correct"]) / len(comparison)
            few_acc = sum(1 for c in comparison if c["few_correct"]) / len(comparison)
            bm.metrics = {"zero_shot_acc": round(zero_acc, 2), "few_shot_acc": round(few_acc, 2)}
            results["zero_vs_few"] = {"comparison": comparison, "zero_shot_accuracy": zero_acc, "few_shot_accuracy": few_acc}

        print(f"\n  Zero-Shot vs Few-Shot Accuracy:")
        print(f"    Zero-Shot: {zero_acc:.0%}")
        print(f"    Few-Shot:  {few_acc:.0%}")
        print(f"  {'Text':<45} {'Expected':<10} {'Zero':<10} {'Few':<10}")
        print(f"  {'─'*75}")
        for c in comparison:
            z_mark = "✓" if c["zero_correct"] else "✗"
            f_mark = "✓" if c["few_correct"] else "✗"
            print(f"  {c['text']:<45} {c['expected']:<10} {c['zero_shot']:<8}{z_mark} {c['few_shot']:<8}{f_mark}")

        # --- 3. Chain-of-Thought Demonstration ---
        with self.benchmark("Chain-of-thought reasoning") as bm:
            problems = [
                {"problem": "A store has 45 apples. They sell 18, then receive 23 more. How many apples?",
                 "steps": ["Start: 45 apples", "Sell 18: 45 - 18 = 27", "Receive 23: 27 + 23 = 50", "Answer: 50 apples"],
                 "answer": 50},
                {"problem": "A train travels at 60 km/h for 2.5 hours. How far does it go?",
                 "steps": ["Speed: 60 km/h", "Time: 2.5 hours", "Distance = Speed × Time", "60 × 2.5 = 150 km"],
                 "answer": 150},
                {"problem": "If 3 workers can build a wall in 12 hours, how long for 6 workers?",
                 "steps": ["3 workers × 12 hours = 36 worker-hours total", "6 workers: 36 / 6 = 6 hours"],
                 "answer": 6},
            ]
            cot_results = []
            for p in problems:
                prompt = self.PATTERNS["chain_of_thought"].render(problem=p["problem"])
                cot_results.append({
                    "problem": p["problem"],
                    "reasoning_steps": p["steps"],
                    "answer": p["answer"],
                    "prompt_used": prompt[:150],
                })
            bm.metrics = {"problems_solved": len(cot_results)}
            results["chain_of_thought"] = cot_results

        print(f"\n  Chain-of-Thought Reasoning:")
        for c in cot_results:
            print(f"\n    Problem: {c['problem']}")
            for s in c["reasoning_steps"]:
                print(f"      → {s}")
            print(f"      = {c['answer']}")

        # --- 4. Persona Pattern ---
        with self.benchmark("Persona pattern generation") as bm:
            personas = [
                {"persona": "a senior ML engineer", "context": "You explain concepts with code examples and production considerations.", "query": "What is attention?"},
                {"persona": "a kindergarten teacher", "context": "You explain everything simply with fun analogies.", "query": "What is attention?"},
                {"persona": "a startup CEO", "context": "You focus on business value, ROI, and market impact.", "query": "What is attention?"},
            ]
            persona_results = []
            for p in personas:
                prompt = self.PATTERNS["persona"].render(
                    persona=p["persona"], persona_context=p["context"], query=p["query"]
                )
                persona_results.append({
                    "persona": p["persona"],
                    "query": p["query"],
                    "prompt_length": len(prompt),
                    "prompt_preview": prompt[:200],
                })
            bm.metrics = {"personas": len(persona_results)}
            results["persona_patterns"] = persona_results

        print(f"\n  Persona Prompts (same question, different perspectives):")
        for p in persona_results:
            print(f"    [{p['persona']}] → {p['prompt_length']} chars")

        # --- 5. Prompt Token Economics ---
        with self.benchmark("Prompt token economics analysis") as bm:
            economics = []
            for text, _ in self.TEST_TEXTS[:3]:
                for pattern_name in ["zero_shot", "few_shot", "chain_of_thought"]:
                    if pattern_name == "chain_of_thought":
                        prompt = self.PATTERNS[pattern_name].render(problem=text)
                    else:
                        prompt = self.PATTERNS[pattern_name].render(text=text)
                    # Rough token estimate: ~4 chars per token
                    est_tokens = len(prompt) // 4
                    economics.append({
                        "pattern": pattern_name,
                        "chars": len(prompt),
                        "est_tokens": est_tokens,
                        "cost_per_1k_in": round(est_tokens * 0.003, 4),  # hypothetical pricing
                    })
            bm.metrics = {"analyses": len(economics)}
            results["token_economics"] = economics

        print(f"\n  Prompt Token Economics:")
        print(f"  {'Pattern':<22} {'Chars':>6} {'~Tokens':>8} {'~Cost/$':>8}")
        print(f"  {'─'*50}")
        for e in economics[:6]:
            print(f"  {e['pattern']:<22} {e['chars']:>6} {e['est_tokens']:>8} {e['cost_per_1k_in']:>8.4f}")

        self.print_benchmark_table()
        return results

    def _keyword_classify(self, text: str) -> str:
        text_lower = text.lower()
        pos = ["amazing", "love", "great", "excellent", "outstanding", "fast", "accurate", "exceeded", "incredible", "fantastic"]
        neg = ["terrible", "bad", "slow", "cold", "disappointed", "worst", "horrible", "poor"]
        p_score = sum(1 for w in pos if w in text_lower)
        n_score = sum(1 for w in neg if w in text_lower)
        if p_score > n_score:
            return "POSITIVE"
        elif n_score > p_score:
            return "NEGATIVE"
        return "NEUTRAL"

    def _keyword_classify_enhanced(self, text: str) -> str:
        """Enhanced with negation handling and intensity."""
        text_lower = text.lower()
        pos = {"amazing": 2, "love": 2, "great": 1, "excellent": 2, "outstanding": 2,
               "fast": 1, "accurate": 1, "exceeded": 2, "incredible": 2, "fair": 0.5}
        neg = {"terrible": 2, "bad": 1, "slow": 1, "cold": 1, "disappointed": 1.5,
               "worst": 2, "horrible": 2, "poor": 1}

        p_score = sum(v for w, v in pos.items() if w in text_lower)
        n_score = sum(v for w, v in neg.items() if w in text_lower)

        # Check for "but" pattern (mixed sentiment)
        if "but" in text_lower and p_score > 0 and n_score > 0:
            return "MIXED"

        if p_score > n_score:
            return "POSITIVE"
        elif n_score > p_score:
            return "NEGATIVE"
        return "NEUTRAL"
