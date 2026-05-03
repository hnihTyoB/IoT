"""
Training & Validation Loops for IoT Device Identification
===========================================================
Three training modes:

1. Pretrain (ET-BERT style): Masked Feature Modeling
   - No labels needed
   - Transformer learns to reconstruct masked flow features
   - Output: pre-trained encoder weights

2. Contrastive (AOC-IDS style): Self-Supervised Contrastive Learning
   - Uses device labels to form positive/negative pairs
   - Transformer learns behavioral embeddings
   - Output: encoder that produces separable embeddings

3. Finetune: Supervised Classification
   - Uses pre-trained encoder + classification head
   - Standard cross-entropy training
   - Output: full classifier model

Adapted from bandwidth-estimation's training_validation_loop.py
"""

import copy
import math
from typing import Tuple, List, Dict, Literal, Optional, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm.auto import tqdm

from .models import (
    IoTTransformerEncoder,
    IoTDeviceClassifier,
    MaskedFeatureModeling,
    ProjectionHead,
    iot_collate_fn,
    iot_contrastive_collate_fn,
)
from .losses import CRCLoss, NTXentLoss, TripletLoss


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Pre-training with Masked Feature Modeling (ET-BERT style)
# ──────────────────────────────────────────────────────────────────────────────
def pretrain_masked(
    model: MaskedFeatureModeling,
    train_ds,
    val_ds,
    num_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[MaskedFeatureModeling, List[Dict[str, float]]]:
    """
    Pre-train using Masked Feature Modeling (ET-BERT-style).
    
    The model learns to reconstruct masked flow features from context.
    This teaches the Transformer to understand IoT traffic behavioral patterns
    without any device labels.
    """
    model.to(device)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=iot_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=iot_collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    history = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        # ── Train ──
        model.train()
        train_loss_sum, train_count = 0.0, 0

        pbar = tqdm(train_loader, desc=f"[Pretrain] Epoch {epoch}/{num_epochs}", leave=False)
        for features, _ in pbar:
            features = features.to(device)
            optimizer.zero_grad()

            loss, _ = model(features)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * features.size(0)
            train_count += features.size(0)
            pbar.set_postfix({"loss": loss.item()})

        train_loss = train_loss_sum / max(train_count, 1)

        # ── Validate ──
        model.eval()
        val_loss_sum, val_count = 0.0, 0

        with torch.no_grad():
            for features, _ in val_loader:
                features = features.to(device)
                loss, _ = model(features)
                val_loss_sum += loss.item() * features.size(0)
                val_count += features.size(0)

        val_loss = val_loss_sum / max(val_count, 1)

        print(
            f"  Epoch {epoch:>2}/{num_epochs} | "
            f"train_recon_loss: {train_loss:.6f} | val_recon_loss: {val_loss:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Contrastive Learning (AOC-IDS style)
# ──────────────────────────────────────────────────────────────────────────────
def train_contrastive(
    encoder: IoTTransformerEncoder,
    train_ds,
    val_ds,
    num_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 5e-4,
    temperature: float = 0.1,
    loss_type: Literal["ntxent", "triplet", "crc"] = "ntxent",
    device: str = "cpu",
    proj_dim: int = 64,
) -> Tuple[IoTTransformerEncoder, List[Dict[str, float]]]:
    """
    Contrastive Learning phase (AOC-IDS style) with ProjectionHead.

    Trains the encoder to produce embeddings where same-device flows
    cluster together and different-device flows are pushed apart.

    Uses a ProjectionHead (SimCLR-style) to map encoder outputs to a
    lower-dimensional space for contrastive loss. After training, the
    projection head is discarded — only the encoder weights are kept.

    Uses the IoTContrastiveDataset which provides (anchor, positive, negative) triplets.
    """
    encoder.to(device)

    # Create ProjectionHead (SimCLR-style, learned from IOT-DETECTOR)
    projection_head = ProjectionHead(
        d_model=encoder.d_model, proj_dim=proj_dim
    ).to(device)
    print(f"  [ProjectionHead] {encoder.d_model}D → {proj_dim}D (discarded after training)")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=iot_contrastive_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=iot_contrastive_collate_fn,
    )

    # Select loss function
    if loss_type == "ntxent":
        criterion = NTXentLoss(temperature=temperature)
    elif loss_type == "triplet":
        criterion = TripletLoss(margin=1.0)
    elif loss_type == "crc":
        criterion = CRCLoss(temperature=temperature)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Optimize both encoder and projection head
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    history = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        # ── Train ──
        encoder.train()
        projection_head.train()
        train_loss_sum, train_count = 0.0, 0

        pbar = tqdm(train_loader, desc=f"[Contrastive] Epoch {epoch}/{num_epochs}", leave=False)
        for anchor, positive, negative, labels in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Get embeddings from encoder, then project for contrastive loss
            anchor_emb = projection_head(encoder(anchor))
            positive_emb = projection_head(encoder(positive))
            negative_emb = projection_head(encoder(negative))

            if loss_type == "crc":
                # CRC uses all embeddings + labels
                all_emb = torch.cat([anchor_emb, positive_emb, negative_emb], dim=0)
                # Labels: anchor and positive share same label, negative gets different
                # For simplicity, use anchor labels for CRC
                all_labels = torch.cat([labels, labels, labels + 1000], dim=0)
                loss = criterion(all_emb, all_labels)
            else:
                loss = criterion(anchor_emb, positive_emb, negative_emb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * anchor.size(0)
            train_count += anchor.size(0)
            pbar.set_postfix({"loss": loss.item()})

        train_loss = train_loss_sum / max(train_count, 1)

        # ── Validate ──
        encoder.eval()
        projection_head.eval()
        val_loss_sum, val_count = 0.0, 0

        with torch.no_grad():
            for anchor, positive, negative, labels in val_loader:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                labels = labels.to(device)

                anchor_emb = projection_head(encoder(anchor))
                positive_emb = projection_head(encoder(positive))
                negative_emb = projection_head(encoder(negative))

                if loss_type == "crc":
                    all_emb = torch.cat([anchor_emb, positive_emb, negative_emb], dim=0)
                    all_labels = torch.cat([labels, labels, labels + 1000], dim=0)
                    loss = criterion(all_emb, all_labels)
                else:
                    loss = criterion(anchor_emb, positive_emb, negative_emb)

                val_loss_sum += loss.item() * anchor.size(0)
                val_count += anchor.size(0)

        val_loss = val_loss_sum / max(val_count, 1)

        print(
            f"  Epoch {epoch:>2}/{num_epochs} | "
            f"train_contrastive_loss: {train_loss:.6f} | val_contrastive_loss: {val_loss:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(encoder.state_dict())

        scheduler.step()

    if best_state is not None:
        encoder.load_state_dict(best_state)

    return encoder, history


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Supervised Fine-tuning (Classification)
# ──────────────────────────────────────────────────────────────────────────────
def finetune_classifier(
    model: IoTDeviceClassifier,
    train_ds,
    val_ds,
    num_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cpu",
    freeze_encoder: bool = False,
) -> Tuple[IoTDeviceClassifier, Dict[str, Any], List[Dict[str, float]]]:
    """
    Fine-tune the full classifier (encoder + classification head).

    If freeze_encoder=True, only the classification head is trained.
    Uses pre-trained encoder weights from Phase 1/2.
    """
    model.to(device)

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("[Freeze] Encoder parameters frozen, training only classifier head.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=iot_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=iot_collate_fn,
    )

    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    history = []
    best_f1 = 0.0
    best_state = None
    best_stats = {}

    for epoch in range(1, num_epochs + 1):
        # ── Train ──
        model.train()
        train_loss_sum, train_count = 0.0, 0

        pbar = tqdm(train_loader, desc=f"[Finetune] Epoch {epoch}/{num_epochs}", leave=False)
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * features.size(0)
            train_count += features.size(0)
            pbar.set_postfix({"loss": loss.item()})

        train_loss = train_loss_sum / max(train_count, 1)

        # ── Validate ──
        model.eval()
        val_loss_sum, val_count = 0.0, 0
        all_y_true, all_y_pred = [], []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                logits = model(features)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * features.size(0)
                val_count += features.size(0)

                preds = logits.argmax(dim=1)
                all_y_true.append(labels.cpu().numpy())
                all_y_pred.append(preds.cpu().numpy())

        val_loss = val_loss_sum / max(val_count, 1)
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(
            f"  Epoch {epoch:>2}/{num_epochs} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"accuracy: {acc:.4f} | macro_f1: {f1:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "macro_f1": f1,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            best_stats = {
                "best_epoch": epoch,
                "best_f1": best_f1,
                "best_accuracy": acc,
                "best_val_loss": val_loss,
                "y_true": y_true,
                "y_pred": y_pred,
            }

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_stats, history
