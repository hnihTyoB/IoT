from scipy.spatial.distance import cosine
import torch
import pandas as pd
import numpy as np
import json
import joblib
import yaml
import warnings
from pathlib import Path
from typing import Dict, Optional

from functions_and_modules.models import IoTDeviceClassifier
from functions_and_modules.dataset import UNSW_NUMERIC_FEATURES

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Fallback threshold khi không có per-device stats
FALLBACK_THRESHOLD = 0.12
# Temperature cho Softmax calibration
TEMPERATURE = 2.0


class IoTInferenceEngine:
    def __init__(self, experiment_dir: str, device: str = "cpu"):
        self.exp_dir = Path(experiment_dir)
        self.device = torch.device(device)
        
        # 1. Load config và nhãn
        with open(self.exp_dir / "label_map.json", "r") as f:
            self.label_map = json.load(f)
        self.inv_label_map = {int(v): k for k, v in self.label_map.items()}
        
        with open(self.exp_dir / "config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
            
        # 2. Load bộ cân bằng dữ liệu
        self.scaler = joblib.load(self.exp_dir / "scaler_features.pkl")
        self.feature_names = self.config.get("features", UNSW_NUMERIC_FEATURES)
        
        # 3. Khởi tạo mô hình
        self.model = IoTDeviceClassifier(
            input_dim=len(self.feature_names),
            num_classes=len(self.label_map),
            d_model=self.config.get("d_model", 128),
            nhead=self.config.get("nhead", 4),
            num_layers=self.config.get("num_layers", 4)
        ).to(self.device)
        
        ckpt_path = self.exp_dir / "model.pt"
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

        # 4. Load centroids cho anomaly detection
        centroid_path = self.exp_dir / "centroids.pt"
        self.centroids = torch.load(centroid_path, weights_only=False) if centroid_path.exists() else None

        # 5. Load per-device distance stats
        stats_path = self.exp_dir / "device_dist_stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                self.device_dist_stats = json.load(f)
            print(f"  ✅ Loaded per-device thresholds cho {len(self.device_dist_stats)} thiết bị")
        else:
            self.device_dist_stats = None
            print(f"  ⚠️ Không tìm thấy device_dist_stats.json, dùng threshold cố định = {FALLBACK_THRESHOLD}")

    def _get_threshold(self, device_name: str) -> float:
        """Lấy adaptive threshold cho từng thiết bị (mean + 2σ)."""
        if self.device_dist_stats and device_name in self.device_dist_stats:
            return self.device_dist_stats[device_name]["threshold"]
        return FALLBACK_THRESHOLD

    def preprocess_flows(self, df: pd.DataFrame) -> torch.Tensor:
        # Trích xuất các đặc trưng cần thiết
        data_df = df[self.feature_names]
        
        # Scale dữ liệu
        data_scaled = self.scaler.transform(data_df)
        
        # Windowing
        window_size = self.config.get("window_size", 10)
        if len(data_scaled) < window_size:
            pad = np.zeros((window_size - len(data_scaled), data_scaled.shape[1]))
            data_scaled = np.vstack([data_scaled, pad])
            
        window = data_scaled[:window_size]
        return torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)

    def predict(self, csv_path: str) -> Dict:
        df = pd.read_csv(csv_path)
        
        # Điền các đặc trưng thiếu bằng 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
                
        input_tensor = self.preprocess_flows(df)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            
            # Temperature Scaling — hiệu chỉnh Softmax overconfident
            calibrated_logits = logits / TEMPERATURE
            probs = torch.softmax(calibrated_logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            conf, pred_idx = conf.item(), pred_idx.item()
            pred_name = self.inv_label_map.get(pred_idx, "Unknown")

            # Lấy vector đặc trưng (Embedding) - Encoder đã tự động Pooling rồi
            emb_tensor = self.model.encoder(input_tensor)  # (1, 128)
            embedding = emb_tensor[0].cpu().numpy()  # (128,)
            
            # Tính khoảng cách hành vi + Per-device Threshold
            dist = 999.0
            is_unknown = True
            device_threshold = FALLBACK_THRESHOLD
            
            if self.centroids and pred_name in self.centroids:
                v_emb = embedding.flatten()
                v_centroid = np.asarray(self.centroids[pred_name]).flatten()
                dist = float(cosine(v_emb, v_centroid))
                
                # Per-device adaptive threshold
                device_threshold = self._get_threshold(pred_name)
                is_unknown = (dist > device_threshold)
            else:
                is_unknown = True

            if is_unknown:
                return {
                    "device": "UNKNOWN DEVICE",
                    "confidence": conf,
                    "distance": dist,
                    "threshold": device_threshold,
                    "status": "CẢNH BÁO: Thiết bị lạ hoặc traffic bất thường!",
                    "nearest_device": pred_name,
                }
            return {
                "device": pred_name,
                "confidence": conf,
                "distance": dist,
                "threshold": device_threshold,
                "status": "ĐÃ XÁC MINH",
                "nearest_device": pred_name,
            }


def format_result(res: Dict) -> str:
    lines = []
    lines.append("─" * 55)
    
    if res["device"] == "UNKNOWN DEVICE":
        lines.append(f"  KẾT QUẢ NHẬN DIỆN : UNKNOWN DEVICE")
        lines.append(f"  THIẾT BỊ GẦN NHẤT  : {res['nearest_device']}")
    else:
        lines.append(f"  KẾT QUẢ NHẬN DIỆN : {res['device']}")
    
    lines.append(f"  ĐỘ TIN CẬY        : {res['confidence']:.4f}")
    lines.append(f"  KHOẢNG CÁCH        : {res['distance']:.4f}  (ngưỡng: {res['threshold']:.4f})")
    lines.append(f"  TRẠNG THÁI         : {res['status']}")
    lines.append("─" * 55)
    return "\n".join(lines)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HỆ THỐNG NHẬN DIỆN THIẾT BỊ IOT - SELF-SUPERVISED LEARNING")
    print("=" * 60)

    BASE_DIR = Path(__file__).resolve().parent
    
    BEST_EXP = BASE_DIR / "experiments" / "finetune_robust_frozen"
    if not BEST_EXP.exists():
        BEST_EXP = BASE_DIR.parent / "experiments" / "finetune_robust_frozen"
    if not BEST_EXP.exists():
        print(f"Error: Không tìm thấy thư mục mô hình tại {BEST_EXP}")
    else:
        engine = IoTInferenceEngine(BEST_EXP)
        
        while True:
            file_input = input("\nNhập đường dẫn file CSV traffic (hoặc 'q' để thoát): ").strip()
            
            if file_input.lower() == 'q':
                break
                
            test_file = Path(file_input)
            if not test_file.exists():
                print(f"File không tồn tại! Hãy thử lại.")
                continue
                
            print(f"Đang phân tích: {test_file.name}...")
            res = engine.predict(str(test_file))
            print(format_result(res))

    print("\nCảm ơn bạn đã sử dụng hệ thống!")
