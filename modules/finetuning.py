"""
Module 9: Fine-Tuning Concepts
=================================
Demonstrates fine-tuning workflow: data preparation, training loop simulation,
loss curves, checkpoint management, and evaluation metrics.
"""

import math
import random
import time
import json
import os
from .base import BaseModule


class TrainingDataset:
    """Synthetic training dataset for fine-tuning demonstration."""

    SENTIMENT_DATA = [
        ("The product quality exceeded my expectations", "positive"),
        ("Absolutely terrible customer service experience", "negative"),
        ("It works as described nothing special", "neutral"),
        ("Best purchase I have ever made highly recommend", "positive"),
        ("Waste of money do not buy this product", "negative"),
        ("Average performance for the price point", "neutral"),
        ("Outstanding build quality and fast delivery", "positive"),
        ("Broke after two days very disappointed", "negative"),
        ("It is okay but I expected more features", "neutral"),
        ("Five stars perfect in every way", "positive"),
        ("Would give zero stars if possible", "negative"),
        ("Decent product acceptable quality", "neutral"),
        ("Love it amazing design and functionality", "positive"),
        ("Horrible experience will never order again", "negative"),
        ("Standard product does what it says", "neutral"),
        ("Incredible value for money must buy", "positive"),
        ("Total disaster avoid at all costs", "negative"),
        ("Neither good nor bad just mediocre", "neutral"),
        ("Superb quality craftsmanship is excellent", "positive"),
        ("Regret buying this completely useless", "negative"),
    ]

    def __init__(self):
        self.label_map = {"positive": 0, "negative": 1, "neutral": 2}

    def get_splits(self, train_ratio: float = 0.8) -> tuple:
        random.seed(42)
        data = list(self.SENTIMENT_DATA)
        random.shuffle(data)
        split = int(len(data) * train_ratio)
        return data[:split], data[split:]


class SimpleClassifier:
    """A from-scratch trainable text classifier (no external ML libs needed)."""

    def __init__(self, vocab_size: int, num_classes: int, hidden_dim: int = 32):
        random.seed(42)
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Initialize weights (simplified)
        self.embedding = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(vocab_size)]
        self.weights = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(num_classes)]
        self.bias = [0.0] * num_classes

    def forward(self, token_ids: list[int]) -> list[float]:
        """Forward pass: embed → average → linear → softmax."""
        if not token_ids:
            return [1.0 / self.num_classes] * self.num_classes

        # Average embedding
        avg = [0.0] * self.hidden_dim
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                for d in range(self.hidden_dim):
                    avg[d] += self.embedding[tid][d]
        for d in range(self.hidden_dim):
            avg[d] /= len(token_ids)

        # Linear layer
        logits = []
        for c in range(self.num_classes):
            logit = self.bias[c]
            for d in range(self.hidden_dim):
                logit += avg[d] * self.weights[c][d]
            logits.append(logit)

        # Softmax
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_logits)
        return [e / total for e in exp_logits]

    def compute_loss(self, probs: list[float], target: int) -> float:
        """Cross-entropy loss."""
        return -math.log(max(probs[target], 1e-8))

    def train_step(self, token_ids: list[int], target: int, lr: float = 0.01) -> float:
        """Single training step with numerical gradient approximation."""
        probs = self.forward(token_ids)
        loss = self.compute_loss(probs, target)

        # Simplified gradient update (perturbation-based)
        eps = 1e-4
        for c in range(self.num_classes):
            for d in range(min(self.hidden_dim, 16)):  # Update subset for speed
                original = self.weights[c][d]
                self.weights[c][d] = original + eps
                loss_plus = self.compute_loss(self.forward(token_ids), target)
                self.weights[c][d] = original
                grad = (loss_plus - loss) / eps
                self.weights[c][d] -= lr * grad

        return loss

    def predict(self, token_ids: list[int]) -> int:
        probs = self.forward(token_ids)
        return probs.index(max(probs))


class FineTuningModule(BaseModule):
    name = "Fine-Tuning Concepts"
    description = "Data prep, training loop, loss curves, checkpoints, evaluation metrics"

    def run(self) -> dict:
        self.print_header()
        results = {}

        # --- 1. Data Preparation ---
        with self.benchmark("Prepare training dataset") as bm:
            dataset = TrainingDataset()
            train_data, val_data = dataset.get_splits(0.8)

            # Build vocabulary
            vocab = {}
            for text, _ in dataset.SENTIMENT_DATA:
                for word in text.lower().split():
                    if word not in vocab:
                        vocab[word] = len(vocab)

            data_info = {
                "total_samples": len(dataset.SENTIMENT_DATA),
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "vocab_size": len(vocab),
                "num_classes": len(dataset.label_map),
                "class_distribution": {},
            }
            for _, label in train_data:
                data_info["class_distribution"][label] = data_info["class_distribution"].get(label, 0) + 1

            bm.metrics = {"train": len(train_data), "val": len(val_data), "vocab": len(vocab)}
            results["data_preparation"] = data_info

        print(f"\n  Dataset: {data_info['total_samples']} samples → {data_info['train_samples']} train / {data_info['val_samples']} val")
        print(f"  Vocab: {data_info['vocab_size']} words, {data_info['num_classes']} classes")
        print(f"  Distribution: {data_info['class_distribution']}")

        # --- 2. Training Loop ---
        classifier = SimpleClassifier(vocab_size=len(vocab), num_classes=3)
        num_epochs = 15
        training_history = []

        with self.benchmark(f"Training loop ({num_epochs} epochs)") as bm:
            for epoch in range(num_epochs):
                epoch_loss = 0
                random.shuffle(train_data)
                for text, label in train_data:
                    token_ids = [vocab.get(w, 0) for w in text.lower().split()]
                    target = dataset.label_map[label]
                    loss = classifier.train_step(token_ids, target, lr=0.005)
                    epoch_loss += loss

                avg_loss = epoch_loss / len(train_data)

                # Validation
                correct = 0
                for text, label in val_data:
                    token_ids = [vocab.get(w, 0) for w in text.lower().split()]
                    pred = classifier.predict(token_ids)
                    if pred == dataset.label_map[label]:
                        correct += 1
                val_acc = correct / len(val_data)

                training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": round(avg_loss, 4),
                    "val_accuracy": round(val_acc, 4),
                })

            bm.metrics = {"final_loss": training_history[-1]["train_loss"], "final_acc": training_history[-1]["val_accuracy"]}
            results["training_history"] = training_history

        print(f"\n  Training Progress:")
        print(f"  {'Epoch':>6} {'Loss':>8} {'Val Acc':>9}  {'Loss Curve'}")
        print(f"  {'─'*55}")
        for h in training_history:
            bar_loss = "█" * max(1, int(h["train_loss"] * 10))
            bar_acc = "▓" * int(h["val_accuracy"] * 20)
            print(f"  {h['epoch']:>6} {h['train_loss']:>8.4f} {h['val_accuracy']:>9.4f}  {bar_loss}")

        # --- 3. Checkpoint Management ---
        with self.benchmark("Checkpoint management") as bm:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "checkpoints")
            os.makedirs(output_dir, exist_ok=True)
            checkpoints = []
            for h in training_history:
                if h["epoch"] % 5 == 0 or h["epoch"] == num_epochs:
                    ckpt = {
                        "epoch": h["epoch"],
                        "loss": h["train_loss"],
                        "accuracy": h["val_accuracy"],
                        "path": f"checkpoints/model_epoch_{h['epoch']}.pt",
                    }
                    checkpoints.append(ckpt)
                    # Save mock checkpoint
                    ckpt_path = os.path.join(output_dir, f"model_epoch_{h['epoch']}.json")
                    with open(ckpt_path, "w") as f:
                        json.dump(ckpt, f)

            bm.metrics = {"checkpoints_saved": len(checkpoints)}
            results["checkpoints"] = checkpoints

        print(f"\n  Checkpoints saved: {len(checkpoints)}")
        for c in checkpoints:
            print(f"    Epoch {c['epoch']}: loss={c['loss']:.4f}, acc={c['accuracy']:.4f} → {c['path']}")

        # --- 4. Evaluation Metrics ---
        with self.benchmark("Compute evaluation metrics") as bm:
            # Confusion matrix
            confusion = [[0] * 3 for _ in range(3)]
            predictions = []
            for text, label in val_data:
                token_ids = [vocab.get(w, 0) for w in text.lower().split()]
                pred = classifier.predict(token_ids)
                true = dataset.label_map[label]
                confusion[true][pred] += 1
                predictions.append({"text": text[:40], "true": label, "pred": list(dataset.label_map.keys())[pred],
                                    "correct": pred == true})

            # Per-class metrics
            class_names = list(dataset.label_map.keys())
            per_class = {}
            for i, name in enumerate(class_names):
                tp = confusion[i][i]
                fp = sum(confusion[j][i] for j in range(3)) - tp
                fn = sum(confusion[i][j] for j in range(3)) - tp
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                per_class[name] = {
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "support": sum(confusion[i]),
                }

            overall_acc = sum(confusion[i][i] for i in range(3)) / max(sum(sum(r) for r in confusion), 1)
            eval_metrics = {
                "accuracy": round(overall_acc, 4),
                "confusion_matrix": confusion,
                "per_class": per_class,
                "predictions": predictions,
            }
            bm.metrics = {"accuracy": round(overall_acc, 4)}
            results["evaluation"] = eval_metrics

        print(f"\n  Evaluation Metrics:")
        print(f"    Overall Accuracy: {overall_acc:.4f}")
        print(f"\n    Confusion Matrix:")
        print(f"    {'':>10}", end="")
        for n in class_names:
            print(f" {n:>8}", end="")
        print()
        for i, name in enumerate(class_names):
            print(f"    {name:>10}", end="")
            for j in range(3):
                print(f" {confusion[i][j]:>8}", end="")
            print()
        print(f"\n    {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print(f"    {'─'*42}")
        for name, m in per_class.items():
            print(f"    {name:<12} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")

        # --- 5. Learning rate comparison ---
        with self.benchmark("Learning rate comparison study") as bm:
            lr_study = []
            for lr in [0.001, 0.005, 0.01, 0.05]:
                test_clf = SimpleClassifier(vocab_size=len(vocab), num_classes=3)
                for epoch in range(5):
                    for text, label in train_data:
                        token_ids = [vocab.get(w, 0) for w in text.lower().split()]
                        test_clf.train_step(token_ids, dataset.label_map[label], lr=lr)
                # Evaluate
                correct = sum(1 for text, label in val_data
                              if test_clf.predict([vocab.get(w, 0) for w in text.lower().split()]) == dataset.label_map[label])
                lr_study.append({"learning_rate": lr, "accuracy_5_epochs": round(correct / len(val_data), 4)})
            bm.metrics = {"lr_configs": len(lr_study)}
            results["lr_study"] = lr_study

        print(f"\n  Learning Rate Study (5 epochs each):")
        for s in lr_study:
            bar = "█" * int(s["accuracy_5_epochs"] * 20)
            print(f"    lr={s['learning_rate']:<6} acc={s['accuracy_5_epochs']:.4f} {bar}")

        self.print_benchmark_table()
        return results
