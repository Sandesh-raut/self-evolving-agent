"""
Module 8: Multi-Modal Processing
===================================
Demonstrates text-image alignment, cross-modal embeddings,
image feature extraction, and multi-modal fusion strategies.
"""

import math
import struct
import zlib
import os
from .base import BaseModule


class ImageFeatureExtractor:
    """Extract statistical features from images (works without PIL for basic analysis)."""

    def analyze_raw_bytes(self, image_bytes: bytes) -> dict:
        """Analyze raw image bytes for basic statistics."""
        byte_values = list(image_bytes[:1024])  # Sample first 1KB
        if not byte_values:
            return {"error": "empty image"}

        return {
            "size_bytes": len(image_bytes),
            "byte_entropy": self._entropy(byte_values),
            "mean_byte_value": round(sum(byte_values) / len(byte_values), 2),
            "std_byte_value": round(self._std(byte_values), 2),
            "unique_bytes": len(set(byte_values)),
            "format_signature": image_bytes[:4].hex() if len(image_bytes) >= 4 else "unknown",
        }

    def _entropy(self, values: list[int]) -> float:
        from collections import Counter
        counts = Counter(values)
        total = len(values)
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 4)

    def _std(self, values: list[int]) -> float:
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)


class SyntheticImageGenerator:
    """Generate minimal PNG images for testing (pure Python, no deps)."""

    @staticmethod
    def create_png(width: int, height: int, color: tuple[int, int, int] = (100, 150, 200)) -> bytes:
        """Create a simple solid-color PNG image."""

        def chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

        # PNG signature
        signature = b'\x89PNG\r\n\x1a\n'

        # IHDR
        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
        ihdr = chunk(b'IHDR', ihdr_data)

        # IDAT - raw pixel data
        raw_data = b''
        for y in range(height):
            raw_data += b'\x00'  # Filter: None
            for x in range(width):
                raw_data += bytes(color)

        compressed = zlib.compress(raw_data)
        idat = chunk(b'IDAT', compressed)

        # IEND
        iend = chunk(b'IEND', b'')

        return signature + ihdr + idat + iend


class TextImageAligner:
    """Align text descriptions with image features (simplified CLIP-like concept)."""

    def __init__(self):
        self.text_features_cache = {}
        self.image_features_cache = {}

    def extract_text_features(self, text: str) -> list[float]:
        """Extract feature vector from text (simplified)."""
        import re
        words = re.findall(r'\w+', text.lower())
        # Map words to features via hash-based projection
        dim = 64
        features = [0.0] * dim
        for word in words:
            h = hash(word)
            for d in range(dim):
                features[d] += ((h >> d) & 1) * 2 - 1
        # Normalize
        norm = math.sqrt(sum(f * f for f in features)) + 1e-8
        return [f / norm for f in features]

    def extract_image_features(self, image_stats: dict) -> list[float]:
        """Extract feature vector from image statistics."""
        dim = 64
        features = [0.0] * dim
        # Map image stats to feature dimensions
        values = [
            image_stats.get("byte_entropy", 0),
            image_stats.get("mean_byte_value", 0) / 255,
            image_stats.get("std_byte_value", 0) / 128,
            image_stats.get("unique_bytes", 0) / 256,
            image_stats.get("size_bytes", 0) / 100000,
        ]
        for i, v in enumerate(values):
            for d in range(dim):
                features[d] += v * math.sin((d + 1) * (i + 1) * 0.1)
        # Normalize
        norm = math.sqrt(sum(f * f for f in features)) + 1e-8
        return [f / norm for f in features]

    def compute_alignment(self, text_features: list[float], image_features: list[float]) -> float:
        """Compute text-image alignment score (cosine similarity)."""
        dot = sum(a * b for a, b in zip(text_features, image_features))
        return max(0, min(1, (dot + 1) / 2))  # Scale to [0, 1]


class MultiModalModule(BaseModule):
    name = "Multi-Modal Processing"
    description = "Text-image alignment, cross-modal embeddings, feature extraction, fusion"

    IMAGE_DESCRIPTIONS = [
        ("A bright sunny beach with blue water", (100, 180, 230)),
        ("A dark forest at night", (20, 40, 15)),
        ("A red sports car on a highway", (200, 50, 30)),
        ("A green meadow with flowers", (80, 180, 60)),
        ("A snowy mountain peak", (230, 235, 240)),
    ]

    def run(self) -> dict:
        self.print_header()
        results = {}

        generator = SyntheticImageGenerator()
        extractor = ImageFeatureExtractor()
        aligner = TextImageAligner()

        # --- 1. Synthetic Image Generation ---
        with self.benchmark("Generate synthetic test images") as bm:
            images = []
            for desc, color in self.IMAGE_DESCRIPTIONS:
                png_data = generator.create_png(32, 32, color)
                images.append({"description": desc, "color": color, "data": png_data, "size": len(png_data)})
            bm.metrics = {"images_generated": len(images), "total_bytes": sum(i["size"] for i in images)}

        print(f"\n  Generated {len(images)} synthetic test images")

        # --- 2. Image Feature Extraction ---
        image_analyses = []
        with self.benchmark("Extract image features") as bm:
            for img in images:
                analysis = extractor.analyze_raw_bytes(img["data"])
                analysis["description"] = img["description"]
                analysis["color_rgb"] = img["color"]
                image_analyses.append(analysis)
            bm.metrics = {"images_analyzed": len(image_analyses)}
            results["image_features"] = image_analyses

        print(f"\n  Image Feature Extraction:")
        print(f"  {'Description':<40} {'Entropy':>8} {'Mean':>6} {'Unique':>7}")
        print(f"  {'─'*65}")
        for a in image_analyses:
            print(f"  {a['description']:<40} {a['byte_entropy']:>8.4f} {a['mean_byte_value']:>6.1f} {a['unique_bytes']:>7}")

        # --- 3. Text-Image Alignment ---
        with self.benchmark("Compute text-image alignment matrix") as bm:
            alignment_matrix = []
            for i, img in enumerate(image_analyses):
                text_feat = aligner.extract_text_features(img["description"])
                img_feat = aligner.extract_image_features(img)
                row = []
                for j, other_img in enumerate(image_analyses):
                    other_img_feat = aligner.extract_image_features(other_img)
                    score = aligner.compute_alignment(text_feat, other_img_feat)
                    row.append(round(score, 4))
                alignment_matrix.append(row)

            # Check if diagonal has highest scores (correct alignment)
            correct = sum(1 for i in range(len(alignment_matrix))
                         if alignment_matrix[i][i] == max(alignment_matrix[i]))
            accuracy = correct / len(alignment_matrix)
            bm.metrics = {"alignment_accuracy": round(accuracy, 2)}
            results["alignment_matrix"] = alignment_matrix
            results["alignment_accuracy"] = accuracy

        print(f"\n  Text-Image Alignment Matrix (higher = better match):")
        print(f"  {'':>5}", end="")
        for i in range(len(alignment_matrix)):
            print(f"  Img{i}", end="")
        print()
        for i, row in enumerate(alignment_matrix):
            print(f"  T{i}: ", end="")
            for j, val in enumerate(row):
                marker = "*" if i == j else " "
                print(f" {val:.3f}{marker}", end="")
            print()
        print(f"  (* = expected match, accuracy = {accuracy:.0%})")

        # --- 4. Cross-Modal Retrieval ---
        with self.benchmark("Cross-modal retrieval test") as bm:
            retrieval_queries = [
                "ocean and beach scene",
                "dark and mysterious nature",
                "fast vehicle on road",
                "peaceful nature with greenery",
            ]
            retrieval_results = []
            for query in retrieval_queries:
                q_feat = aligner.extract_text_features(query)
                scored = []
                for img in image_analyses:
                    img_feat = aligner.extract_image_features(img)
                    score = aligner.compute_alignment(q_feat, img_feat)
                    scored.append({"description": img["description"], "score": round(score, 4)})
                scored.sort(key=lambda x: x["score"], reverse=True)
                retrieval_results.append({"query": query, "top_match": scored[0], "all_matches": scored})
            bm.metrics = {"queries": len(retrieval_queries)}
            results["cross_modal_retrieval"] = retrieval_results

        print(f"\n  Cross-Modal Retrieval:")
        for r in retrieval_results:
            print(f"    Query: '{r['query']}'")
            print(f"    → Best: '{r['top_match']['description']}' (score={r['top_match']['score']:.4f})")

        # --- 5. Multi-Modal Fusion ---
        with self.benchmark("Multi-modal feature fusion") as bm:
            fusion_results = []
            for img in image_analyses:
                text_feat = aligner.extract_text_features(img["description"])
                img_feat = aligner.extract_image_features(img)

                # Early fusion: concatenate
                early = text_feat + img_feat
                # Late fusion: average
                late = [(t + i) / 2 for t, i in zip(text_feat, img_feat)]
                # Attention-weighted fusion
                weights = [abs(t) / (abs(t) + abs(i) + 1e-8) for t, i in zip(text_feat, img_feat)]
                attention = [t * w + i * (1 - w) for t, i, w in zip(text_feat, img_feat, weights)]

                fusion_results.append({
                    "description": img["description"][:30],
                    "early_dim": len(early),
                    "late_dim": len(late),
                    "attention_dim": len(attention),
                    "early_norm": round(math.sqrt(sum(x*x for x in early)), 4),
                    "late_norm": round(math.sqrt(sum(x*x for x in late)), 4),
                    "attention_norm": round(math.sqrt(sum(x*x for x in attention)), 4),
                })

            bm.metrics = {"fusion_strategies": 3}
            results["fusion"] = fusion_results

        print(f"\n  Multi-Modal Fusion Comparison:")
        print(f"  {'Description':<32} {'Early':>10} {'Late':>10} {'Attention':>10}")
        print(f"  {'─'*65}")
        for f in fusion_results:
            print(f"  {f['description']:<32} {f['early_dim']:>5}d/{f['early_norm']:.2f} "
                  f"{f['late_dim']:>4}d/{f['late_norm']:.2f} "
                  f"{f['attention_dim']:>4}d/{f['attention_norm']:.2f}")

        # --- 6. Save test images ---
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        with self.benchmark("Save test images to disk") as bm:
            saved = 0
            for i, img in enumerate(images):
                path = os.path.join(output_dir, f"test_image_{i}.png")
                with open(path, "wb") as f:
                    f.write(img["data"])
                saved += 1
            bm.metrics = {"images_saved": saved}

        print(f"\n  Saved {saved} test images to outputs/")

        self.print_benchmark_table()
        return results
