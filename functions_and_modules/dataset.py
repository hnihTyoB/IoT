"""
UNSW-IoTraffic Flow Sequence Dataset
=====================================
Reads per-device CSV flow files, groups flows into fixed-length windows,
and returns (features, device_label) pairs for Transformer-based models.

Adapted from bandwidth-estimation's FlowSequenceDataset with:
  - Device label extraction from filenames
  - Fixed-length windowing (sliding window) instead of per-client grouping
  - Numeric feature selection matching UNSW-IoTraffic CSV columns
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from tqdm.auto import tqdm
import re


# ─────────────────────────────────────────────────────────────────────────────
# Feature columns from UNSW-IoTraffic flows CSV
# ─────────────────────────────────────────────────────────────────────────────
UNSW_NUMERIC_FEATURES = [
    "srcNumPackets",
    "dstNumPackets",
    "srcPayloadSize",
    "dstPayloadSize",
    "srcAvgPayloadSize",
    "dstAvgPayloadSize",
    "srcMaxPayloadSize",
    "dstMaxPayloadSize",
    "srcStdDevPayloadSize",
    "dstStdDevPayloadSize",
    "flowDuration",
    "srcAvgInterarrivalTime",
    "dstAvgInterarrivalTime",
    "avgInterarrivalTime",
    "srcStdDevInterarrivalTime",
    "dstStdDevInterarrivalTime",
    "stdDevInterarrivalTime",
]


def extract_device_name(filename: str) -> str:
    """
    Extract device name from UNSW-IoTraffic CSV filenames.
    Example: 'AmazonEcho_44650d56ccd3_flows.csv' -> 'AmazonEcho'
    """
    # Pattern: DeviceName_MACaddr_flows.csv
    match = re.match(r"^([A-Za-z]+(?:[A-Za-z0-9]*))_", filename)
    if match:
        return match.group(1)
    return filename.split("_")[0]


def load_unsw_flows(
    data_dir: str,
    feature_names: Optional[List[str]] = None,
    min_flows_per_device: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load all UNSW-IoTraffic CSV flow files from a directory.
    
    Returns
    -------
    df : pd.DataFrame
        Combined dataframe with 'device_name' and 'device_label' columns.
    label_map : dict
        Mapping from device name to integer label.
    """
    if feature_names is None:
        feature_names = UNSW_NUMERIC_FEATURES

    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*_flows.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No flow CSV files found in {data_dir}")

    dfs = []
    for csv_path in tqdm(csv_files, desc="Loading UNSW flow CSVs"):
        device_name = extract_device_name(csv_path.name)
        df = pd.read_csv(csv_path, low_memory=False)

        # Keep only numeric features that exist
        available_features = [f for f in feature_names if f in df.columns]
        if len(available_features) < len(feature_names) // 2:
            print(f"  Warning: {csv_path.name} missing many features, skipping.")
            continue

        df = df[available_features].copy()

        # Convert all feature columns to numeric, coerce errors
        for col in available_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["device_name"] = device_name
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Fill NaNs with 0
    for col in feature_names:
        if col in df_all.columns:
            df_all[col] = df_all[col].fillna(0.0)

    # Filter devices with too few flows
    device_counts = df_all["device_name"].value_counts()
    valid_devices = device_counts[device_counts >= min_flows_per_device].index.tolist()
    df_all = df_all[df_all["device_name"].isin(valid_devices)].reset_index(drop=True)

    # Create label encoding
    le = LabelEncoder()
    df_all["device_label"] = le.fit_transform(df_all["device_name"])
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))

    print(f"\nLoaded {len(df_all)} flows from {len(label_map)} devices:")
    for name, label in sorted(label_map.items(), key=lambda x: x[1]):
        count = (df_all["device_label"] == label).sum()
        print(f"  [{label:>2}] {name}: {count:,} flows")

    return df_all, label_map


class IoTFlowWindowDataset(Dataset):
    """
    Sliding-window dataset over IoT flow sequences.

    Each sample is a window of `window_size` consecutive flows from the same
    device, labeled with the device's integer ID.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain feature columns + 'device_name' + 'device_label'.
    feature_names : list of str
        Numeric feature column names.
    window_size : int
        Number of consecutive flows per sample.
    stride : int
        Step between windows (1 = max overlap, window_size = no overlap).
    feature_scaler : StandardScaler or None
        If None, fits a new scaler on this data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        window_size: int = 10,
        stride: int = 5,
        feature_scaler: Optional[StandardScaler] = None,
    ):
        self.feature_names = feature_names
        self.window_size = window_size
        self.stride = stride

        # ── Scale features ──────────────────────────────────────────────
        available = [f for f in feature_names if f in df.columns]
        self.feature_scaler = feature_scaler or StandardScaler()

        if feature_scaler is None:
            df[available] = self.feature_scaler.fit_transform(df[available])
        else:
            df[available] = self.feature_scaler.transform(df[available])

        df[available] = df[available].fillna(0.0)

        # ── Build windows per device ────────────────────────────────────
        self.windows: List[Tuple[np.ndarray, int]] = []

        for device_label, group in tqdm(
            df.groupby("device_label"), desc="Building flow windows"
        ):
            features = group[available].values.astype(np.float32)
            n_flows = len(features)

            if n_flows < window_size:
                # Pad short sequences
                pad = np.zeros((window_size - n_flows, len(available)), dtype=np.float32)
                features = np.vstack([features, pad])
                self.windows.append((features, int(device_label)))
            else:
                for start in range(0, n_flows - window_size + 1, stride):
                    window = features[start : start + window_size]
                    self.windows.append((window, int(device_label)))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        features, label = self.windows[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def get_feature_scaler(self) -> StandardScaler:
        return self.feature_scaler


class IoTContrastiveDataset(Dataset):
    """
    Contrastive Learning dataset for IoT device identification.

    For each anchor window, returns:
      - anchor: flow window from device A
      - positive: different flow window from the SAME device A
      - negative: flow window from a DIFFERENT device B

    This directly implements the Contrastive Learning paradigm from AOC-IDS,
    adapted to use flow windows instead of raw features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        window_size: int = 10,
        stride: int = 5,
        feature_scaler: Optional[StandardScaler] = None,
        num_pairs_per_device: int = 500,
    ):
        self.feature_names = feature_names
        self.window_size = window_size

        available = [f for f in feature_names if f in df.columns]
        self.feature_scaler = feature_scaler or StandardScaler()

        if feature_scaler is None:
            df[available] = self.feature_scaler.fit_transform(df[available])
        else:
            df[available] = self.feature_scaler.transform(df[available])

        df[available] = df[available].fillna(0.0)

        # ── Build per-device window pools ───────────────────────────────
        self.device_windows: Dict[int, List[np.ndarray]] = {}
        self.all_devices: List[int] = []

        for device_label, group in tqdm(
            df.groupby("device_label"), desc="Building contrastive windows"
        ):
            features = group[available].values.astype(np.float32)
            n_flows = len(features)
            windows = []

            if n_flows < window_size:
                pad = np.zeros((window_size - n_flows, len(available)), dtype=np.float32)
                features = np.vstack([features, pad])
                windows.append(features)
            else:
                for start in range(0, n_flows - window_size + 1, stride):
                    windows.append(features[start : start + window_size])

            if len(windows) >= 2:  # Need at least 2 for positive pairs
                self.device_windows[int(device_label)] = windows
                self.all_devices.append(int(device_label))

        # ── Build triplet indices ───────────────────────────────────────
        self.triplets: List[Tuple[int, int, int, int, int]] = []
        # (anchor_device, anchor_idx, positive_idx, negative_device, negative_idx)

        rng = np.random.RandomState(42)
        for device in self.all_devices:
            n_windows = len(self.device_windows[device])
            other_devices = [d for d in self.all_devices if d != device]
            if not other_devices:
                continue

            for _ in range(min(num_pairs_per_device, n_windows)):
                a_idx, p_idx = rng.choice(n_windows, size=2, replace=(n_windows < 2))
                if a_idx == p_idx and n_windows > 1:
                    p_idx = (a_idx + 1) % n_windows

                neg_device = rng.choice(other_devices)
                n_idx = rng.randint(len(self.device_windows[neg_device]))

                self.triplets.append((device, a_idx, p_idx, neg_device, n_idx))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        dev_a, a_idx, p_idx, dev_n, n_idx = self.triplets[idx]

        anchor = torch.tensor(self.device_windows[dev_a][a_idx], dtype=torch.float32)
        positive = torch.tensor(self.device_windows[dev_a][p_idx], dtype=torch.float32)
        negative = torch.tensor(self.device_windows[dev_n][n_idx], dtype=torch.float32)
        label = torch.tensor(dev_a, dtype=torch.long)

        return anchor, positive, negative, label

    def get_feature_scaler(self) -> StandardScaler:
        return self.feature_scaler
