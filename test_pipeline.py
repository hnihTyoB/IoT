"""
Quick end-to-end test of the IoT Device Identification pipeline.
Tests: data loading -> windowing -> all 3 training phases (1 epoch each).
"""
import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import numpy as np
from sklearn.model_selection import train_test_split

from functions_and_modules.dataset import (
    load_unsw_flows,
    IoTFlowWindowDataset,
    IoTContrastiveDataset,
    UNSW_NUMERIC_FEATURES,
)
from functions_and_modules.models import (
    IoTTransformerEncoder,
    IoTDeviceClassifier,
    MaskedFeatureModeling,
    iot_collate_fn,
)
from functions_and_modules.training import (
    pretrain_masked,
    train_contrastive,
    finetune_classifier,
)
from functions_and_modules.visualization import extract_embeddings


def main():
    DATA_DIR = "../dataset"
    WINDOW_SIZE = 10
    STRIDE = 10  # non-overlapping for speed
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    EPOCHS = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Step 1: Load UNSW data ──────────────────────────────────────────
    print("=" * 60)
    print("  STEP 1: Loading UNSW-IoTraffic flows")
    print("=" * 60)
    df, label_map = load_unsw_flows(
        DATA_DIR,
        feature_names=UNSW_NUMERIC_FEATURES,
        min_flows_per_device=100,
    )
    num_classes = len(label_map)
    num_features = len(UNSW_NUMERIC_FEATURES)
    print(f"\nTotal flows: {len(df):,}")
    print(f"Devices: {num_classes}")
    print(f"Features: {num_features}")

    # Split
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["device_label"], random_state=42
    )

    # ── Step 2: Build Windowed Datasets ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  STEP 2: Building flow window datasets")
    print("=" * 60)
    train_ds = IoTFlowWindowDataset(
        train_df, UNSW_NUMERIC_FEATURES,
        window_size=WINDOW_SIZE, stride=STRIDE,
    )
    val_ds = IoTFlowWindowDataset(
        val_df, UNSW_NUMERIC_FEATURES,
        window_size=WINDOW_SIZE, stride=STRIDE,
        feature_scaler=train_ds.get_feature_scaler(),
    )
    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows: {len(val_ds)}")

    # Quick data check
    feat, label = train_ds[0]
    print(f"Sample shape: features={feat.shape}, label={label.item()}")

    # ── Step 3: Phase 1 - Masked Feature Modeling ───────────────────────
    print(f"\n{'=' * 60}")
    print("  STEP 3: Phase 1 - Masked Feature Modeling (ET-BERT)")
    print("=" * 60)
    mlm_model = MaskedFeatureModeling(
        input_dim=num_features, d_model=D_MODEL,
        nhead=NHEAD, num_layers=NUM_LAYERS, mask_ratio=0.15,
    )
    mlm_model, mlm_history = pretrain_masked(
        mlm_model, train_ds, val_ds,
        num_epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3, device=device,
    )
    print(f"  Final MLM loss: {mlm_history[-1]['val_loss']:.6f}")

    # ── Step 4: Phase 2 - Contrastive Learning ──────────────────────────
    print(f"\n{'=' * 60}")
    print("  STEP 4: Phase 2 - Contrastive Learning (AOC-IDS)")
    print("=" * 60)
    # Build contrastive dataset (fewer pairs for speed)
    train_contrastive_ds = IoTContrastiveDataset(
        train_df, UNSW_NUMERIC_FEATURES,
        window_size=WINDOW_SIZE, stride=STRIDE,
        num_pairs_per_device=50,
    )
    val_contrastive_ds = IoTContrastiveDataset(
        val_df, UNSW_NUMERIC_FEATURES,
        window_size=WINDOW_SIZE, stride=STRIDE,
        feature_scaler=train_contrastive_ds.get_feature_scaler(),
        num_pairs_per_device=20,
    )
    print(f"Train triplets: {len(train_contrastive_ds)}")
    print(f"Val triplets: {len(val_contrastive_ds)}")

    # Load pre-trained encoder from MLM
    encoder = IoTTransformerEncoder(
        input_dim=num_features, d_model=D_MODEL,
        nhead=NHEAD, num_layers=NUM_LAYERS,
    )
    encoder.load_state_dict(mlm_model.encoder.state_dict())
    print("  Loaded MLM pre-trained encoder weights")

    encoder, cl_history = train_contrastive(
        encoder, train_contrastive_ds, val_contrastive_ds,
        num_epochs=EPOCHS, batch_size=BATCH_SIZE, lr=5e-4,
        loss_type="ntxent", device=device,
    )
    print(f"  Final contrastive loss: {cl_history[-1]['val_loss']:.6f}")

    # ── Step 5: Phase 3 - Fine-tuning ───────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  STEP 5: Phase 3 - Supervised Fine-tuning")
    print("=" * 60)
    classifier = IoTDeviceClassifier(
        input_dim=num_features, num_classes=num_classes,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
    )
    # Load contrastive encoder weights
    classifier.encoder.load_state_dict(encoder.state_dict())
    print("  Loaded contrastive pre-trained encoder weights")

    classifier, best_stats, ft_history = finetune_classifier(
        classifier, train_ds, val_ds,
        num_epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-4, device=device,
    )
    print(f"  Best accuracy: {best_stats.get('best_accuracy', 0):.4f}")
    print(f"  Best F1: {best_stats.get('best_f1', 0):.4f}")

    # ── Step 6: Extract Embeddings ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  STEP 6: Extracting embeddings for t-SNE")
    print("=" * 60)
    embeddings, labels = extract_embeddings(classifier, val_ds, device=device)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique devices in embeddings: {len(np.unique(labels))}")

    # ── Done ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  [OK] ALL PIPELINE STAGES PASSED!")
    print("=" * 60)
    print(f"\n  Summary:")
    print(f"    Devices:       {num_classes}")
    print(f"    Features:      {num_features}")
    print(f"    Windows:       {len(train_ds)} train / {len(val_ds)} val")
    print(f"    MLM Loss:      {mlm_history[-1]['val_loss']:.6f}")
    print(f"    CL Loss:       {cl_history[-1]['val_loss']:.6f}")
    print(f"    Accuracy:      {best_stats.get('best_accuracy', 0):.4f}")
    print(f"    Macro F1:      {best_stats.get('best_f1', 0):.4f}")
    print(f"    Embedding dim: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
