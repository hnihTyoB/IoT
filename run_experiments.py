"""
run_experiments.py — Main Entry Point for IoT Device Identification
====================================================================
Orchestrates the full training pipeline:

  Phase 1 (pretrain):     Masked Feature Modeling (ET-BERT style)
  Phase 2 (contrastive):  Self-Supervised Contrastive Learning (AOC-IDS style)
  Phase 3 (finetune):     Supervised Device Classification

Usage:
  python run_experiments.py --config configs/pretrain_unsw.yaml
  python run_experiments.py --config configs/contrastive_unsw.yaml
  python run_experiments.py --config configs/finetune_unsw.yaml

Adapted from bandwidth-estimation's run_experiments.py
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from functions_and_modules.dataset import (
    IoTFlowWindowDataset,
    IoTContrastiveDataset,
    load_unsw_flows,
    UNSW_NUMERIC_FEATURES,
)
from functions_and_modules.models import (
    IoTTransformerEncoder,
    IoTDeviceClassifier,
    MaskedFeatureModeling,
)
from functions_and_modules.training import (
    pretrain_masked,
    train_contrastive,
    finetune_classifier,
)
from functions_and_modules.visualization import (
    plot_tsne_embeddings,
    plot_training_curves,
    plot_confusion_matrix,
    extract_embeddings,
    generate_classification_report,
)
from functions_and_modules.experiment_artifacts import save_experiment_artifacts


# ──────────────────────────────────────────────────────────────────────────────
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base config (shallow)."""
    merged = dict(base)
    merged.update(overrides)
    return merged


def detect_device() -> str:
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon).")
        return "mps"
    if torch.cuda.is_available():
        print("Using CUDA.")
        return "cuda"
    print("Using CPU.")
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Pre-training (Masked Feature Modeling)
# ──────────────────────────────────────────────────────────────────────────────
def run_pretrain(cfg: dict) -> dict:
    """ET-BERT-style pre-training with Masked Feature Modeling."""
    exp_id = cfg["experiment_id"]
    out_dir = Path("experiments") / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    device = detect_device()

    print(f"\n{'='*70}")
    print(f"  PHASE 1: MASKED FEATURE MODELING (ET-BERT Style)")
    print(f"  Experiment: {exp_id}")
    print(f"{'='*70}\n")

    # Load data
    feature_names = cfg.get("features", UNSW_NUMERIC_FEATURES)
    df, label_map = load_unsw_flows(
        cfg["data_dir"],
        feature_names=feature_names,
        min_flows_per_device=cfg.get("min_flows_per_device", 100),
    )

    # Train/Val split by device
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["device_label"], random_state=42
    )

    # Build datasets (windowed)
    train_ds = IoTFlowWindowDataset(
        train_df, feature_names,
        window_size=cfg.get("window_size", 10),
        stride=cfg.get("stride", 5),
    )
    val_ds = IoTFlowWindowDataset(
        val_df, feature_names,
        window_size=cfg.get("window_size", 10),
        stride=cfg.get("stride", 5),
        feature_scaler=train_ds.get_feature_scaler(),
    )

    print(f"\nTrain windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    # Build MLM model
    model = MaskedFeatureModeling(
        input_dim=len(feature_names),
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 4),
        dropout=cfg.get("dropout", 0.1),
        mask_ratio=cfg.get("mask_ratio", 0.15),
    )

    # Train
    model, history = pretrain_masked(
        model, train_ds, val_ds,
        num_epochs=cfg.get("num_epochs", 20),
        batch_size=cfg.get("batch_size", 64),
        lr=float(cfg.get("lr", 1e-3)),
        device=device,
    )

    # Save
    plot_training_curves(history, save_path=str(out_dir / "training_curves.png"),
                         title=f"Pre-training: {exp_id}")

    # Extract and visualize embeddings from the encoder
    embeddings, labels = extract_embeddings(model.encoder, val_ds, device=device)
    plot_tsne_embeddings(embeddings, labels, label_map,
                         save_path=str(out_dir / "tsne_pretrain.png"),
                         title=f"t-SNE after Pre-training: {exp_id}")

    save_experiment_artifacts(
        output_dir=out_dir,
        config=cfg,
        model=model,
        metrics=history,
        feature_scaler=train_ds.get_feature_scaler(),
        label_map=label_map,
    )

    return {"experiment_id": exp_id, "output_dir": str(out_dir), "mode": "pretrain"}


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Contrastive Learning
# ──────────────────────────────────────────────────────────────────────────────
def run_contrastive(cfg: dict) -> dict:
    """AOC-IDS-style contrastive learning for behavioral embeddings."""
    exp_id = cfg["experiment_id"]
    out_dir = Path("experiments") / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    device = detect_device()

    print(f"\n{'='*70}")
    print(f"  PHASE 2: CONTRASTIVE LEARNING (AOC-IDS Style)")
    print(f"  Experiment: {exp_id}")
    print(f"{'='*70}\n")

    # Load data
    feature_names = cfg.get("features", UNSW_NUMERIC_FEATURES)
    df, label_map = load_unsw_flows(
        cfg["data_dir"],
        feature_names=feature_names,
        min_flows_per_device=cfg.get("min_flows_per_device", 100),
    )

    # Train/Val split
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["device_label"], random_state=42
    )

    # Build contrastive datasets
    train_ds = IoTContrastiveDataset(
        train_df, feature_names,
        window_size=cfg.get("window_size", 10),
        stride=cfg.get("stride", 5),
        num_pairs_per_device=cfg.get("num_pairs_per_device", 500),
    )
    val_ds = IoTContrastiveDataset(
        val_df, feature_names,
        window_size=cfg.get("window_size", 10),
        stride=cfg.get("stride", 5),
        feature_scaler=train_ds.get_feature_scaler(),
        num_pairs_per_device=cfg.get("num_pairs_per_device", 200),
    )

    print(f"\nTrain triplets: {len(train_ds)}, Val triplets: {len(val_ds)}")

    # Build encoder
    encoder = IoTTransformerEncoder(
        input_dim=len(feature_names),
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 4),
        dropout=cfg.get("dropout", 0.1),
    )

    # Load pre-trained weights if available
    pre_id = cfg.get("pretrain_experiment_id")
    if pre_id:
        ckpt_path = Path("experiments") / pre_id / "model.pt"
        if ckpt_path.exists():
            # Load only encoder weights from MLM model
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            encoder_state = {
                k.replace("encoder.", ""): v
                for k, v in state.items()
                if k.startswith("encoder.")
            }
            encoder.load_state_dict(encoder_state, strict=False)
            print(f"[Loaded] Pre-trained encoder from {ckpt_path}")
        else:
            print(f"[Warning] Checkpoint not found: {ckpt_path}, training from scratch.")

    # Train
    encoder, history = train_contrastive(
        encoder, train_ds, val_ds,
        num_epochs=cfg.get("num_epochs", 30),
        batch_size=cfg.get("batch_size", 64),
        lr=float(cfg.get("lr", 5e-4)),
        temperature=cfg.get("temperature", 0.1),
        loss_type=cfg.get("loss_type", "ntxent"),
        device=device,
    )

    # Visualize
    plot_training_curves(history, save_path=str(out_dir / "training_curves.png"),
                         title=f"Contrastive Training: {exp_id}")

    # t-SNE on windowed dataset for visualization
    window_val_ds = IoTFlowWindowDataset(
        val_df, feature_names,
        window_size=cfg.get("window_size", 10),
        stride=cfg.get("stride", 5),
        feature_scaler=train_ds.get_feature_scaler(),
    )
    embeddings, labels = extract_embeddings(encoder, window_val_ds, device=device)
    plot_tsne_embeddings(embeddings, labels, label_map,
                         save_path=str(out_dir / "tsne_contrastive.png"),
                         title=f"t-SNE after Contrastive Learning: {exp_id}")

    save_experiment_artifacts(
        output_dir=out_dir,
        config=cfg,
        model=encoder,
        metrics=history,
        feature_scaler=train_ds.get_feature_scaler(),
        label_map=label_map,
    )

    return {"experiment_id": exp_id, "output_dir": str(out_dir), "mode": "contrastive"}


def calculate_and_save_centroids(model, dataloader, out_dir, device):
    """Tính toán và lưu Centroids ngay sau khi training xong."""
    model.eval()
    centroids_sum = {}
    centroids_count = {}
    
    print(f"\n🎯 Đang tự động tạo bản đồ hành vi (Centroids) cho {len(dataloader.dataset.label_map)} thiết bị...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating centroids"):
            x = batch["features"].to(device)
            y = batch["labels"].to(device)
            
            # Lấy embedding từ encoder thông qua hàm get_embedding (Pooling trung bình)
            # Dùng thẳng encoder(x).mean(dim=1) vì chúng ta biết cấu trúc model
            embeddings = model.encoder(x).mean(dim=1) 
            
            for i in range(len(y)):
                label = y[i].item()
                if label not in centroids_sum:
                    centroids_sum[label] = torch.zeros_like(embeddings[i])
                    centroids_count[label] = 0
                centroids_sum[label] += embeddings[i]
                centroids_count[label] += 1
                
    # Tính trung bình và chuyển về tên thiết bị
    final_centroids = {}
    inv_label_map = {v: k for k, v in dataloader.dataset.label_map.items()}
    
    for label, total_sum in centroids_sum.items():
        device_name = inv_label_map[label]
        final_centroids[device_name] = (total_sum / centroids_count[label]).cpu().numpy()
        
    save_path = out_dir / "centroids.pt"
    torch.save(final_centroids, save_path)
    print(f"✅ Đã lưu bản đồ hành vi tại: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Fine-tuning (Supervised Classification)
# ──────────────────────────────────────────────────────────────────────────────
def run_finetune(cfg: dict) -> dict:
    """Supervised fine-tuning for device classification."""
    exp_id = cfg["experiment_id"]
    out_dir = Path("experiments") / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    device = detect_device()

    print(f"\n{'='*70}")
    print(f"  PHASE 3: SUPERVISED FINE-TUNING (Classification)")
    print(f"  Experiment: {exp_id}")
    print(f"{'='*70}\n")

    # Load data
    feature_names = cfg.get("features", UNSW_NUMERIC_FEATURES)
    df, label_map = load_unsw_flows(
        cfg["data_dir"],
        feature_names=feature_names,
        min_flows_per_device=cfg.get("min_flows_per_device", 100),
    )

    num_classes = len(label_map)
    print(f"\nNumber of device classes: {num_classes}")

    # Train/Val split
    train_ratio = cfg.get("train_ratio", 0.8)
    train_df, val_df = train_test_split(
        df, test_size=1 - train_ratio, stratify=df["device_label"], random_state=42
    )

    # Build windowed datasets
    train_ds = IoTFlowWindowDataset(
        train_df, feature_names,
        window_size=cfg.get("window_size", 10),
        stride=cfg.get("stride", 5),
    )
    val_ds = IoTFlowWindowDataset(
        val_df, feature_names,
        window_size=cfg.get("window_size", 10),
        stride=cfg.get("stride", 5),
        feature_scaler=train_ds.get_feature_scaler(),
    )

    print(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    # Build classifier model
    model = IoTDeviceClassifier(
        input_dim=len(feature_names),
        num_classes=num_classes,
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 4),
        dropout=cfg.get("dropout", 0.1),
    )

    # Load pre-trained encoder weights if available
    pre_id = cfg.get("pretrain_experiment_id")
    if pre_id:
        ckpt_path = Path("experiments") / pre_id / "model.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            # Try loading as encoder state (from contrastive) or MLM state
            try:
                model.encoder.load_state_dict(state, strict=False)
                print(f"[Loaded] Pre-trained encoder from {ckpt_path}")
            except Exception:
                # Try extracting encoder from MLM model
                encoder_state = {
                    k.replace("encoder.", ""): v
                    for k, v in state.items()
                    if k.startswith("encoder.")
                }
                model.encoder.load_state_dict(encoder_state, strict=False)
                print(f"[Loaded] Pre-trained encoder (MLM) from {ckpt_path}")
        else:
            print(f"[Warning] Checkpoint not found: {ckpt_path}, training from scratch.")

    # Train
    model, best_stats, history = finetune_classifier(
        model, train_ds, val_ds,
        num_epochs=cfg.get("num_epochs", 30),
        batch_size=cfg.get("batch_size", 64),
        lr=float(cfg.get("lr", 1e-4)),
        device=device,
        freeze_encoder=cfg.get("freeze_encoder", False),
    )

    # Visualize results
    plot_training_curves(history, save_path=str(out_dir / "training_curves.png"),
                         title=f"Fine-tuning: {exp_id}")

    if "y_true" in best_stats and "y_pred" in best_stats:
        plot_confusion_matrix(
            best_stats["y_true"], best_stats["y_pred"], label_map,
            save_path=str(out_dir / "confusion_matrix.png"),
        )
        report = generate_classification_report(
            best_stats["y_true"], best_stats["y_pred"], label_map,
            save_path=str(out_dir / "classification_report.txt"),
        )

    # t-SNE visualization of learned embeddings
    embeddings, labels = extract_embeddings(model, val_ds, device=device)
    plot_tsne_embeddings(embeddings, labels, label_map,
                         save_path=str(out_dir / "tsne_finetune.png"),
                         title=f"t-SNE after Fine-tuning: {exp_id}")

    save_experiment_artifacts(
        output_dir=out_dir,
        config=cfg,
        model=model,
        metrics=history,
        model_stats=best_stats,
        feature_scaler=train_ds.get_feature_scaler(),
        label_map=label_map,
    )

    return {
        "experiment_id": exp_id,
        "output_dir": str(out_dir),
        "mode": "finetune",
        "best_f1": best_stats.get("best_f1"),
        "best_accuracy": best_stats.get("best_accuracy"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="IoT Device Identification via Self-Supervised Learning"
    )
    parser.add_argument("--config", required=True, help="YAML config file")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base = cfg["base"]
    runs = list(cfg.get("runs", []))

    if not runs:
        raise ValueError("No runs defined in config.")

    print(f"\n🔬 Launching {len(runs)} experiment(s)...\n")

    runners = {
        "pretrain": run_pretrain,
        "contrastive": run_contrastive,
        "finetune": run_finetune,
    }

    results = []
    for i, run in enumerate(runs, start=1):
        name = run["name"]
        overrides = run.get("overrides", {})
        run_cfg = deep_merge(base, overrides)
        run_cfg["experiment_id"] = run.get("experiment_id", name)
        exp_id = run_cfg["experiment_id"]

        mode = run_cfg.get("mode", "finetune")
        print("-" * 70)
        print(f"  [{i}/{len(runs)}] {exp_id} (mode={mode})")
        print("-" * 70)

        # ─── Run Experiment ───
        runner = runners.get(mode)
        if runner is None:
            print(f"  ⚠ Unknown mode '{mode}', skipping.")
            continue

        summary = runner(run_cfg)
        results.append(summary)

        # ─── Auto-Sync to Google Drive on Colab ───
        import os
        if os.path.exists("/content/drive/MyDrive/"):
            try:
                print(f"  [Auto-Sync] Backing up {exp_id} to Google Drive...")
                os.system(f"cp -r /content/IoT/experiments/{exp_id} /content/drive/MyDrive/experiments/")
            except Exception as e:
                print(f"  [Auto-Sync] Backup failed: {e}")

    print(f"\n{'='*70}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    for s in results:
        info = f"{s['experiment_id']}: {s['mode']} → {s.get('output_dir')}"
        if s.get("best_f1"):
            info += f" | F1={s['best_f1']:.4f}"
        if s.get("best_accuracy"):
            info += f" | Acc={s['best_accuracy']:.4f}"
        print(f"  ✅ {info}")


if __name__ == "__main__":
    main()
