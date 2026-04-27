"""
Module 1: Tokenization & Encoding
==================================
Demonstrates BPE tokenization, subword splitting, vocabulary analysis,
and comparison across different tokenizer implementations.
"""

import re
import time
from collections import Counter
from .base import BaseModule, BenchmarkResult


class BPETokenizer:
    """A from-scratch Byte Pair Encoding tokenizer implementation."""

    def __init__(self):
        self.merges: dict[tuple[str, str], str] = {}
        self.vocab: dict[str, int] = {}

    def _get_pairs(self, tokens: list[str]) -> Counter:
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def train(self, text: str, num_merges: int = 50) -> dict:
        """Train BPE on input text."""
        # Start with character-level tokens
        tokens = list(text)
        merge_log = []

        for i in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0]
            pair, freq = best_pair[0], best_pair[1]
            if freq < 2:
                break

            merged = pair[0] + pair[1]
            self.merges[pair] = merged

            # Apply merge
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j + 1]) == pair:
                    new_tokens.append(merged)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
            merge_log.append({
                "step": i + 1,
                "merged": f"'{pair[0]}' + '{pair[1]}' → '{merged}'",
                "frequency": freq,
                "vocab_size": len(set(tokens)),
            })

        self.vocab = {tok: idx for idx, tok in enumerate(sorted(set(tokens)))}
        return {
            "merges_performed": len(merge_log),
            "final_vocab_size": len(self.vocab),
            "merge_log": merge_log[:10],  # Show first 10
            "sample_vocab": dict(list(self.vocab.items())[:20]),
        }

    def encode(self, text: str) -> list[int]:
        tokens = list(text)
        for pair, merged in self.merges.items():
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j + 1]) == pair:
                    new_tokens.append(merged)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
        return [self.vocab.get(t, -1) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return "".join(reverse_vocab.get(i, "?") for i in ids)


class TokenizationModule(BaseModule):
    name = "Tokenization & Encoding"
    description = "BPE from scratch, HuggingFace tokenizers, subword analysis, compression ratios"

    SAMPLE_TEXTS = [
        "The self-evolving AI agent demonstrates advanced natural language processing capabilities.",
        "Machine learning models use tokenization to convert text into numerical representations.",
        "Transformers revolutionized NLP by introducing self-attention mechanisms for sequence modeling.",
        "आर्टिफिशियल इंटेलिजेंस भविष्य की तकनीक है।",  # Hindi
        "def train_model(data, epochs=10): return model.fit(data, epochs=epochs)",  # Code
    ]

    def run(self) -> dict:
        self.print_header()
        results = {}

        # --- 1. Custom BPE Training ---
        with self.benchmark("BPE Training (from scratch)") as bm:
            corpus = " ".join(self.SAMPLE_TEXTS[:3])
            bpe = BPETokenizer()
            train_result = bpe.train(corpus, num_merges=60)
            bm.metrics = {"vocab_size": train_result["final_vocab_size"], "merges": train_result["merges_performed"]}
            bm.output_preview = str(train_result["merge_log"][:3])
            results["bpe_training"] = train_result

        print(f"\n  Custom BPE Tokenizer trained: {train_result['final_vocab_size']} vocab, {train_result['merges_performed']} merges")
        for m in train_result["merge_log"][:5]:
            print(f"    Step {m['step']}: {m['merged']} (freq={m['frequency']})")

        # --- 2. BPE Encode/Decode ---
        with self.benchmark("BPE Encode + Decode") as bm:
            test_text = "The AI agent evolves"
            encoded = bpe.encode(test_text)
            decoded = bpe.decode(encoded)
            compression = len(test_text) / max(len(encoded), 1)
            bm.metrics = {"compression_ratio": round(compression, 2), "tokens": len(encoded)}
            results["bpe_encode_decode"] = {
                "input": test_text,
                "token_ids": encoded,
                "decoded": decoded,
                "compression_ratio": round(compression, 2),
            }

        print(f"\n  Encode/Decode roundtrip:")
        print(f"    Input:  '{test_text}'")
        print(f"    Tokens: {encoded}")
        print(f"    Decoded: '{decoded}'")
        print(f"    Compression: {compression:.2f}x")

        # --- 3. HuggingFace Tokenizer Comparison ---
        try:
            from transformers import AutoTokenizer

            tokenizer_names = [
                ("bert-base-uncased", "BERT WordPiece"),
                ("gpt2", "GPT-2 BPE"),
                ("t5-small", "T5 SentencePiece"),
            ]

            comparison = []
            for model_name, label in tokenizer_names:
                with self.benchmark(f"HF Tokenizer: {label}") as bm:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    text = self.SAMPLE_TEXTS[0]
                    tokens = tokenizer.tokenize(text)
                    ids = tokenizer.encode(text)
                    decoded = tokenizer.decode(ids)

                    entry = {
                        "tokenizer": label,
                        "model": model_name,
                        "input_chars": len(text),
                        "num_tokens": len(tokens),
                        "tokens_preview": tokens[:10],
                        "vocab_size": tokenizer.vocab_size,
                        "compression_ratio": round(len(text) / len(tokens), 2),
                    }
                    comparison.append(entry)
                    bm.metrics = {"tokens": len(tokens), "vocab_size": tokenizer.vocab_size}

            results["tokenizer_comparison"] = comparison

            print(f"\n  Tokenizer Comparison on: '{self.SAMPLE_TEXTS[0][:60]}...'")
            print(f"  {'Tokenizer':<22} {'Tokens':>7} {'Vocab':>8} {'Ratio':>7}")
            print(f"  {'─'*50}")
            for c in comparison:
                print(f"  {c['tokenizer']:<22} {c['num_tokens']:>7} {c['vocab_size']:>8} {c['compression_ratio']:>7.2f}")

        except ImportError:
            results["tokenizer_comparison"] = "transformers not installed"
            print("  [!] Install `transformers` for HuggingFace tokenizer comparison")

        # --- 4. Multi-language token analysis ---
        with self.benchmark("Multi-language analysis") as bm:
            lang_analysis = []
            for text in self.SAMPLE_TEXTS:
                char_count = len(text)
                bpe_tokens = bpe.encode(text)
                lang_analysis.append({
                    "text_preview": text[:50],
                    "chars": char_count,
                    "bpe_tokens": len(bpe_tokens),
                    "ratio": round(char_count / max(len(bpe_tokens), 1), 2),
                })
            bm.metrics = {"languages_tested": len(self.SAMPLE_TEXTS)}
            results["multilang"] = lang_analysis

        self.print_benchmark_table()
        return results
