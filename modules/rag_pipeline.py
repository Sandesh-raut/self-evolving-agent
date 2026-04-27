"""
Module 3: Retrieval-Augmented Generation (RAG)
================================================
Demonstrates the full RAG pipeline: document chunking, embedding,
retrieval, context assembly, and generation with quality evaluation.
"""

import math
import hashlib
from .base import BaseModule


class DocumentChunker:
    """Splits documents into overlapping chunks for RAG ingestion."""

    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str = "") -> list[dict]:
        words = text.split()
        chunks = []
        start = 0
        idx = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "id": f"{doc_id}_chunk_{idx}",
                "text": chunk_text,
                "start_word": start,
                "end_word": end,
                "word_count": end - start,
            })
            start += self.chunk_size - self.overlap
            idx += 1
        return chunks


class RAGPipeline:
    """Complete RAG pipeline with configurable retrieval and generation."""

    def __init__(self, model=None):
        self.model = model
        self.chunks: list[dict] = []
        self.embeddings: list[list[float]] = []

    def ingest(self, documents: list[dict], chunker: DocumentChunker):
        """Ingest documents: chunk → embed → store."""
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk(doc["content"], doc["id"])
            for chunk in chunks:
                chunk["source"] = doc["title"]
            all_chunks.extend(chunks)
        self.chunks = all_chunks

        # Generate embeddings
        texts = [c["text"] for c in self.chunks]
        if self.model:
            self.embeddings = self.model.encode(texts).tolist()
        else:
            self.embeddings = self._simple_embed(texts)

        return {"chunks_created": len(self.chunks), "avg_chunk_words": sum(c["word_count"] for c in self.chunks) // max(len(self.chunks), 1)}

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve most relevant chunks for a query."""
        if self.model:
            q_emb = self.model.encode(query).tolist()
        else:
            q_emb = self._simple_embed([query])[0]

        scored = []
        for chunk, emb in zip(self.chunks, self.embeddings):
            sim = self._cosine(q_emb, emb)
            scored.append({**chunk, "relevance": round(sim, 4)})
        scored.sort(key=lambda x: x["relevance"], reverse=True)
        return scored[:top_k]

    def generate(self, query: str, context_chunks: list[dict], gen_model=None, gen_tokenizer=None) -> dict:
        """Generate answer from retrieved context."""
        context = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in context_chunks])

        if gen_model and gen_tokenizer:
            import torch
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                output = gen_model.generate(
                    **inputs, max_new_tokens=100, temperature=0.7,
                    do_sample=True, top_p=0.9,
                    pad_token_id=gen_tokenizer.eos_token_id,
                )
            answer = gen_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        else:
            # Extractive fallback: return most relevant sentence from context
            sentences = context.replace("\n", " ").split(". ")
            query_words = set(query.lower().split())
            scored_sents = []
            for s in sentences:
                overlap = len(query_words & set(s.lower().split()))
                scored_sents.append((overlap, s))
            scored_sents.sort(reverse=True)
            answer = scored_sents[0][1] if scored_sents else "No relevant information found."

        # Evaluate answer quality
        faithfulness = self._check_faithfulness(answer, context)

        return {
            "answer": answer,
            "context_length": len(context),
            "chunks_used": len(context_chunks),
            "faithfulness_score": faithfulness,
        }

    def _check_faithfulness(self, answer: str, context: str) -> float:
        """Simple faithfulness check: what fraction of answer words appear in context."""
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        if not answer_words:
            return 0.0
        overlap = len(answer_words & context_words)
        return round(overlap / len(answer_words), 4)

    def _cosine(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-8)

    def _simple_embed(self, texts):
        """TF-IDF fallback."""
        import re
        all_words = set()
        tokenized = []
        for t in texts:
            tokens = re.findall(r'\w+', t.lower())
            tokenized.append(tokens)
            all_words.update(tokens)
        vocab = sorted(all_words)
        idx = {w: i for i, w in enumerate(vocab)}
        result = []
        for tokens in tokenized:
            vec = [0.0] * len(vocab)
            for t in tokens:
                vec[idx[t]] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) + 1e-8
            result.append([x / norm for x in vec])
        return result


class RAGModule(BaseModule):
    name = "RAG Pipeline"
    description = "Document chunking, retrieval, generation, faithfulness evaluation"

    DOCUMENTS = [
        {
            "id": "doc_ml", "title": "Machine Learning Fundamentals",
            "content": (
                "Machine learning is a branch of artificial intelligence that enables systems to learn "
                "and improve from experience without explicit programming. Supervised learning uses labeled "
                "data to train models for classification and regression tasks. Unsupervised learning discovers "
                "hidden patterns in unlabeled data through clustering and dimensionality reduction. "
                "Semi-supervised learning combines small amounts of labeled data with large amounts of "
                "unlabeled data. The bias-variance tradeoff is fundamental: high bias leads to underfitting "
                "while high variance leads to overfitting. Cross-validation helps assess model generalization "
                "by testing on held-out data folds. Feature engineering transforms raw data into informative "
                "representations that improve model performance. Regularization techniques like L1 and L2 "
                "prevent overfitting by penalizing model complexity."
            ),
        },
        {
            "id": "doc_dl", "title": "Deep Learning Architecture",
            "content": (
                "Deep learning uses artificial neural networks with multiple layers to model complex patterns. "
                "Convolutional neural networks excel at image recognition through learned spatial filters. "
                "Recurrent neural networks process sequential data using hidden states that carry information "
                "across time steps. Long short-term memory networks solve the vanishing gradient problem with "
                "gating mechanisms. The transformer architecture replaced RNNs for many tasks by using "
                "self-attention to capture long-range dependencies without sequential processing. "
                "Batch normalization stabilizes training by normalizing layer inputs. Dropout randomly "
                "deactivates neurons during training to prevent co-adaptation. Residual connections in "
                "deep networks allow gradients to flow through skip connections. The Adam optimizer "
                "combines momentum and adaptive learning rates for efficient training."
            ),
        },
        {
            "id": "doc_llm", "title": "Large Language Models",
            "content": (
                "Large language models are transformer-based neural networks trained on vast text corpora. "
                "Pre-training uses self-supervised objectives like next-token prediction or masked language "
                "modeling. Fine-tuning adapts pre-trained models to specific tasks with smaller labeled "
                "datasets. RLHF (Reinforcement Learning from Human Feedback) aligns models with human "
                "preferences. Prompt engineering designs effective instructions to elicit desired model "
                "behavior. In-context learning allows models to perform tasks from examples in the prompt "
                "without parameter updates. Chain-of-thought prompting improves reasoning by encouraging "
                "step-by-step explanations. Retrieval-augmented generation combines LLMs with external "
                "knowledge bases for more accurate and up-to-date responses. Constitutional AI trains "
                "models to be helpful, harmless, and honest through self-critique mechanisms."
            ),
        },
    ]

    QUERIES = [
        "How do transformers work and why are they better than RNNs?",
        "What techniques prevent overfitting in machine learning?",
        "How are large language models trained and aligned?",
        "What is retrieval-augmented generation?",
    ]

    def run(self) -> dict:
        self.print_header()
        results = {}

        # Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            embed_model = None
            print("  [!] Using TF-IDF fallback (install sentence-transformers for better results)")

        # Load generation model
        gen_model, gen_tokenizer = None, None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            gen_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            gen_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            gen_model.eval()
            if gen_tokenizer.pad_token is None:
                gen_tokenizer.pad_token = gen_tokenizer.eos_token
        except ImportError:
            print("  [!] Using extractive fallback (install transformers for generative RAG)")

        # --- 1. Document Chunking ---
        chunker = DocumentChunker(chunk_size=50, overlap=10)
        with self.benchmark("Document chunking") as bm:
            pipeline = RAGPipeline(model=embed_model)
            ingest_result = pipeline.ingest(self.DOCUMENTS, chunker)
            bm.metrics = {"chunks": ingest_result["chunks_created"], "avg_words": ingest_result["avg_chunk_words"]}
            results["ingestion"] = ingest_result

        print(f"\n  Ingested {len(self.DOCUMENTS)} docs → {ingest_result['chunks_created']} chunks (avg {ingest_result['avg_chunk_words']} words)")

        # --- 2. Retrieval + Generation for each query ---
        rag_results = []
        for query in self.QUERIES:
            with self.benchmark(f"RAG: '{query[:35]}...'") as bm:
                retrieved = pipeline.retrieve(query, top_k=3)
                gen_result = pipeline.generate(query, retrieved, gen_model, gen_tokenizer)

                rag_entry = {
                    "query": query,
                    "retrieved_chunks": [{"source": c["source"], "relevance": c["relevance"], "preview": c["text"][:80]} for c in retrieved],
                    "answer": gen_result["answer"][:200],
                    "faithfulness": gen_result["faithfulness_score"],
                }
                rag_results.append(rag_entry)
                bm.metrics = {"faithfulness": gen_result["faithfulness_score"], "top_relevance": retrieved[0]["relevance"] if retrieved else 0}

        results["rag_queries"] = rag_results

        print(f"\n  RAG Query Results:")
        for r in rag_results:
            print(f"\n  Q: {r['query']}")
            print(f"    Top chunk: [{r['retrieved_chunks'][0]['relevance']:.4f}] {r['retrieved_chunks'][0]['source']}")
            print(f"    Answer: {r['answer'][:100]}...")
            print(f"    Faithfulness: {r['faithfulness']:.4f}")

        # --- 3. Chunk size ablation ---
        with self.benchmark("Chunk size ablation study") as bm:
            ablation = []
            for chunk_size in [25, 50, 100, 200]:
                c = DocumentChunker(chunk_size=chunk_size, overlap=chunk_size // 5)
                p = RAGPipeline(model=embed_model)
                ing = p.ingest(self.DOCUMENTS, c)
                retrieved = p.retrieve(self.QUERIES[0], top_k=3)
                gen = p.generate(self.QUERIES[0], retrieved, gen_model, gen_tokenizer)
                ablation.append({
                    "chunk_size": chunk_size,
                    "total_chunks": ing["chunks_created"],
                    "top_relevance": retrieved[0]["relevance"] if retrieved else 0,
                    "faithfulness": gen["faithfulness_score"],
                })
            bm.metrics = {"configs_tested": len(ablation)}
            results["chunk_ablation"] = ablation

        print(f"\n  Chunk Size Ablation:")
        print(f"  {'Size':>6} {'Chunks':>7} {'Top Rel':>9} {'Faith':>8}")
        print(f"  {'─'*35}")
        for a in ablation:
            print(f"  {a['chunk_size']:>6} {a['total_chunks']:>7} {a['top_relevance']:>9.4f} {a['faithfulness']:>8.4f}")

        self.print_benchmark_table()
        return results
