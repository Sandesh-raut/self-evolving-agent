"""
Module 2: Embeddings & Vector Search
======================================
Demonstrates text embeddings, similarity computation, vector indexing,
clustering, and nearest-neighbor search from scratch + with sentence-transformers.
"""

import math
import time
import random
from collections import defaultdict
from .base import BaseModule


class VectorStore:
    """Minimal in-memory vector database with cosine similarity search."""

    def __init__(self):
        self.vectors: list[tuple[str, list[float]]] = []

    def add(self, text: str, embedding: list[float]):
        self.vectors.append((text, embedding))

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        scored = []
        for text, emb in self.vectors:
            sim = self.cosine_similarity(query_embedding, emb)
            scored.append({"text": text, "similarity": round(sim, 4)})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def pairwise_similarity_matrix(self) -> list[list[float]]:
        n = len(self.vectors)
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                sim = self.cosine_similarity(self.vectors[i][1], self.vectors[j][1])
                row.append(round(sim, 4))
            matrix.append(row)
        return matrix


class KMeansClustering:
    """Simple K-Means clustering for embeddings."""

    def __init__(self, k: int = 3, max_iters: int = 20):
        self.k = k
        self.max_iters = max_iters

    def _distance(self, a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def fit(self, embeddings: list[list[float]]) -> dict:
        dim = len(embeddings[0])
        # Initialize centroids randomly
        random.seed(42)
        centroids = random.sample(embeddings, min(self.k, len(embeddings)))

        assignments = [0] * len(embeddings)
        for iteration in range(self.max_iters):
            # Assign points to nearest centroid
            new_assignments = []
            for emb in embeddings:
                dists = [self._distance(emb, c) for c in centroids]
                new_assignments.append(dists.index(min(dists)))

            if new_assignments == assignments:
                break
            assignments = new_assignments

            # Update centroids
            for c in range(self.k):
                members = [embeddings[i] for i in range(len(embeddings)) if assignments[i] == c]
                if members:
                    centroids[c] = [sum(m[d] for m in members) / len(members) for d in range(dim)]

        cluster_sizes = defaultdict(int)
        for a in assignments:
            cluster_sizes[a] += 1

        return {
            "iterations": iteration + 1,
            "k": self.k,
            "assignments": assignments,
            "cluster_sizes": dict(cluster_sizes),
        }


class EmbeddingsModule(BaseModule):
    name = "Embeddings & Vector Search"
    description = "Text embeddings, cosine similarity, vector DB, K-Means clustering, semantic search"

    KNOWLEDGE_BASE = [
        "Machine learning is a subset of artificial intelligence focused on learning from data.",
        "Deep learning uses neural networks with many layers to learn complex patterns.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and analyze visual information.",
        "Reinforcement learning trains agents through rewards and penalties in an environment.",
        "Transfer learning reuses pre-trained models for new tasks with limited data.",
        "Generative AI creates new content including text, images, and code.",
        "Transformer architecture uses self-attention to process sequential data efficiently.",
        "Convolutional neural networks are specialized for processing grid-like data such as images.",
        "Recurrent neural networks process sequences by maintaining hidden state across time steps.",
        "Gradient descent optimizes model parameters by following the steepest descent of the loss.",
        "Backpropagation calculates gradients by applying the chain rule through the network layers.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Embeddings map discrete tokens into continuous vector spaces for semantic similarity.",
        "Fine-tuning adapts pre-trained models to specific downstream tasks.",
    ]

    QUERIES = [
        "How do neural networks learn?",
        "What is the transformer model?",
        "Explain how AI generates images",
        "How does a chatbot understand language?",
    ]

    def run(self) -> dict:
        self.print_header()
        results = {}

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            has_st = True
        except ImportError:
            has_st = False
            print("  [!] sentence-transformers not installed. Using TF-IDF fallback.")

        # --- 1. Generate Embeddings ---
        store = VectorStore()
        with self.benchmark("Generate embeddings for knowledge base") as bm:
            if has_st:
                embeddings = model.encode(self.KNOWLEDGE_BASE).tolist()
            else:
                embeddings = self._tfidf_embeddings(self.KNOWLEDGE_BASE)

            for text, emb in zip(self.KNOWLEDGE_BASE, embeddings):
                store.add(text, emb)

            bm.metrics = {"documents": len(self.KNOWLEDGE_BASE), "dimensions": len(embeddings[0])}

        print(f"\n  Indexed {len(self.KNOWLEDGE_BASE)} documents, {len(embeddings[0])}-dim embeddings")

        # --- 2. Semantic Search ---
        search_results = []
        for query in self.QUERIES:
            with self.benchmark(f"Search: '{query[:30]}...'") as bm:
                if has_st:
                    q_emb = model.encode(query).tolist()
                else:
                    q_emb = self._tfidf_embeddings([query] + self.KNOWLEDGE_BASE)[0]

                hits = store.search(q_emb, top_k=3)
                bm.metrics = {"top_similarity": hits[0]["similarity"] if hits else 0}
                search_results.append({"query": query, "results": hits})

        results["semantic_search"] = search_results

        print(f"\n  Semantic Search Results:")
        for sr in search_results:
            print(f"\n  Q: {sr['query']}")
            for i, hit in enumerate(sr["results"][:2]):
                print(f"    {i+1}. [{hit['similarity']:.4f}] {hit['text'][:70]}...")

        # --- 3. Pairwise Similarity Matrix ---
        with self.benchmark("Pairwise similarity matrix (15x15)") as bm:
            matrix = store.pairwise_similarity_matrix()
            avg_sim = sum(matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix)) if i != j) / (len(matrix) * (len(matrix) - 1))
            bm.metrics = {"avg_pairwise_similarity": round(avg_sim, 4)}
            results["similarity_matrix_stats"] = {
                "size": f"{len(matrix)}x{len(matrix)}",
                "avg_off_diagonal": round(avg_sim, 4),
                "max_off_diagonal": round(max(matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix)) if i != j), 4),
            }

        print(f"\n  Similarity Matrix: {len(matrix)}x{len(matrix)}, avg={avg_sim:.4f}")

        # --- 4. K-Means Clustering ---
        with self.benchmark("K-Means clustering (k=3)") as bm:
            kmeans = KMeansClustering(k=3)
            cluster_result = kmeans.fit(embeddings)
            bm.metrics = {"iterations": cluster_result["iterations"], "clusters": cluster_result["k"]}
            results["clustering"] = cluster_result

        print(f"\n  K-Means Clustering (k=3):")
        print(f"    Converged in {cluster_result['iterations']} iterations")
        print(f"    Cluster sizes: {cluster_result['cluster_sizes']}")
        for cluster_id in sorted(cluster_result["cluster_sizes"].keys()):
            members = [self.KNOWLEDGE_BASE[i][:50] for i, a in enumerate(cluster_result["assignments"]) if a == cluster_id]
            print(f"    Cluster {cluster_id}: {members[:2]}")

        # --- 5. Embedding arithmetic ---
        if has_st:
            with self.benchmark("Embedding arithmetic (analogy)") as bm:
                # king - man + woman ≈ queen (concept demo)
                words = {"king": "king", "man": "man", "woman": "woman", "queen": "queen",
                         "python": "python programming", "java": "java programming",
                         "doctor": "doctor medical", "hospital": "hospital medical"}
                word_embs = {k: model.encode(v).tolist() for k, v in words.items()}

                # king - man + woman should be close to queen
                result_vec = [word_embs["king"][i] - word_embs["man"][i] + word_embs["woman"][i] for i in range(len(word_embs["king"]))]
                sims = {}
                for word, emb in word_embs.items():
                    sims[word] = store.cosine_similarity(result_vec, emb)

                closest = sorted(sims.items(), key=lambda x: x[1], reverse=True)
                bm.metrics = {"closest_to_analogy": closest[0][0]}
                results["embedding_arithmetic"] = {
                    "analogy": "king - man + woman = ?",
                    "similarities": {k: round(v, 4) for k, v in closest},
                }

            print(f"\n  Embedding Arithmetic: king - man + woman = ?")
            for word, sim in closest[:4]:
                print(f"    {word}: {sim:.4f}")

        self.print_benchmark_table()
        return results

    def _tfidf_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Fallback TF-IDF embeddings when sentence-transformers unavailable."""
        import re
        vocab = set()
        tokenized = []
        for text in texts:
            tokens = re.findall(r'\w+', text.lower())
            tokenized.append(tokens)
            vocab.update(tokens)
        vocab = sorted(vocab)
        vocab_idx = {w: i for i, w in enumerate(vocab)}

        embeddings = []
        for tokens in tokenized:
            vec = [0.0] * len(vocab)
            for t in tokens:
                vec[vocab_idx[t]] += 1.0
            # Normalize
            norm = math.sqrt(sum(x*x for x in vec)) + 1e-8
            vec = [x / norm for x in vec]
            embeddings.append(vec)
        return embeddings
