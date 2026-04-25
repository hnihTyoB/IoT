"""
Visualization & Analysis Tools for IoT Device Identification
==============================================================
  - t-SNE embedding visualization (separability analysis)
  - Training curve plots
  - Confusion matrix heatmap
  - Per-device embedding analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Optional
from pathlib import Path
import torch

# Use non-interactive backend for saving
matplotlib.use("Agg")


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_map: Dict[str, int],
    save_path: Optional[str] = None,
    title: str = "t-SNE: IoT Device Behavioral Embeddings",
    perplexity: int = 30,
    figsize: tuple = (14, 10),
) -> None:
    """
    Visualize device embeddings using t-SNE.

    This is the key analysis to demonstrate that the Transformer + SSL model
    learns separable behavioral embeddings for different IoT devices.

    Even devices using the same protocol (e.g., NTP) should form distinct
    clusters based on their unique traffic patterns (packet size, timing, etc.).
    """
    print(f"Computing t-SNE on {embeddings.shape[0]} embeddings...")

    # Subsample if too large to save time
    if embeddings.shape[0] > 5000:
        print("Dataset too large for fast t-SNE. Subsampling 5000 points randomly...")
        np.random.seed(42)
        indices = np.random.choice(embeddings.shape[0], 5000, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    # Reduce dimensionality
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, embeddings.shape[0] // 5)),
        random_state=42,
        n_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    coords = tsne.fit_transform(embeddings)

    # Invert label_map for display
    inv_map = {v: k for k, v in label_map.items()}
    unique_labels = sorted(np.unique(labels))

    # Color palette
    n_colors = len(unique_labels)
    if n_colors <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_colors]
    elif n_colors <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_colors]
    else:
        colors = plt.cm.turbo(np.linspace(0.05, 0.95, n_colors))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = inv_map.get(label, f"Device {label}")
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors[i]],
            label=name,
            s=15,
            alpha=0.7,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=16, fontweight="bold", color="white", pad=15)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12, color="#8b949e")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12, color="#8b949e")
    ax.tick_params(colors="#8b949e")

    for spine in ax.spines.values():
        spine.set_color("#30363d")

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="white",
        title="IoT Devices",
        title_fontsize=10,
    )
    legend.get_title().set_color("white")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        print(f"t-SNE plot saved to {save_path}")
    plt.close()


def plot_training_curves(
    history: List[Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Training Progress",
) -> None:
    """Plot training and validation loss/metrics curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    has_accuracy = "accuracy" in history[0]

    fig, axes = plt.subplots(1, 2 if has_accuracy else 1, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    if not has_accuracy:
        axes = [axes]

    # Loss plot
    ax = axes[0]
    ax.set_facecolor("#0d1117")
    ax.plot(epochs, train_loss, "o-", color="#58a6ff", label="Train Loss", linewidth=2, markersize=4)
    ax.plot(epochs, val_loss, "s-", color="#f78166", label="Val Loss", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch", color="#8b949e")
    ax.set_ylabel("Loss", color="#8b949e")
    ax.set_title("Loss Curves", fontsize=14, fontweight="bold", color="white")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    ax.tick_params(colors="#8b949e")
    ax.grid(True, alpha=0.1, color="#30363d")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    if has_accuracy:
        ax2 = axes[1]
        ax2.set_facecolor("#0d1117")
        accuracy = [h["accuracy"] for h in history]
        f1 = [h["macro_f1"] for h in history]
        ax2.plot(epochs, accuracy, "o-", color="#3fb950", label="Accuracy", linewidth=2, markersize=4)
        ax2.plot(epochs, f1, "s-", color="#d2a8ff", label="Macro F1", linewidth=2, markersize=4)
        ax2.set_xlabel("Epoch", color="#8b949e")
        ax2.set_ylabel("Score", color="#8b949e")
        ax2.set_title("Classification Metrics", fontsize=14, fontweight="bold", color="white")
        ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
        ax2.tick_params(colors="#8b949e")
        ax2.grid(True, alpha=0.1, color="#30363d")
        ax2.set_ylim(0, 1.05)
        for spine in ax2.spines.values():
            spine.set_color("#30363d")

    fig.suptitle(title, fontsize=16, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        print(f"Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Dict[str, int],
    save_path: Optional[str] = None,
    title: str = "Device Classification Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Plot confusion matrix heatmap for device classification."""
    inv_map = {v: k for k, v in label_map.items()}
    labels = sorted(label_map.values())
    names = [inv_map[l] for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
    else:
        cm_normalized = cm

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.6), max(10, len(labels) * 0.5)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="YlOrRd",
        xticklabels=names,
        yticklabels=names,
        ax=ax,
        linewidths=0.5,
        linecolor="#30363d",
        annot_kws={"size": 7},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=15)
    ax.set_xlabel("Predicted Device", fontsize=11, color="#8b949e")
    ax.set_ylabel("True Device", fontsize=11, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def extract_embeddings(
    model,
    dataset,
    device: str = "cpu",
    batch_size: int = 128,
) -> tuple:
    """
    Extract embeddings and labels from a trained model.

    Works with both IoTDeviceClassifier (uses get_embeddings method)
    and IoTTransformerEncoder (uses forward directly).
    """
    from .models import iot_collate_fn

    model.eval()
    model.to(device)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=iot_collate_fn,
    )

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)

            if hasattr(model, "get_embeddings"):
                emb = model.get_embeddings(features)
            elif hasattr(model, "reconstruction_head"): # MaskedFeatureModeling
                emb = model.encoder(features)
            else: # IoTTransformerEncoder
                emb = model(features)

            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return embeddings, labels


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Dict[str, int],
    save_path: Optional[str] = None,
) -> str:
    """Generate and optionally save a detailed classification report."""
    inv_map = {v: k for k, v in label_map.items()}
    labels = sorted(label_map.values())
    names = [inv_map[l] for l in labels]

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=names,
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT — IoT Device Identification")
    print("=" * 70)
    print(report)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write("CLASSIFICATION REPORT — IoT Device Identification\n")
            f.write("=" * 70 + "\n")
            f.write(report)
        print(f"Report saved to {save_path}")

    return report
