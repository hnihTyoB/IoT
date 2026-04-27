from scipy.spatial.distance import cosine
import torch
import pandas as pd
import numpy as np
import json
import joblib
import yaml
import warnings
from pathlib import Path
from typing import Dict

from functions_and_modules.models import IoTDeviceClassifier
from functions_and_modules.dataset import UNSW_NUMERIC_FEATURES

warnings.filterwarnings("ignore", category=UserWarning)
THRESHOLD_DIST = 0.12

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

        centroid_path = self.exp_dir / "centroids.pt"
        self.centroids = torch.load(centroid_path, weights_only=False) if centroid_path.exists() else None

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

    def predict(self, csv_path: str, threshold: float = 0.85) -> Dict:
        df = pd.read_csv(csv_path)
        
        # Điền các đặc trưng thiếu bằng 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
                
        input_tensor = self.preprocess_flows(df)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            conf, pred_idx = conf.item(), pred_idx.item()
            pred_name = self.inv_label_map.get(pred_idx, "Unknown")

            # Lấy vector đặc trưng (Embedding)
            embedding = self.model.encoder(input_tensor).mean(dim=0).cpu().numpy()
            
            # Tính khoảng cách hành vi
            dist = 0.0
            is_unknown = False
            if self.centroids and pred_name in self.centroids:
                dist = cosine(embedding, self.centroids[pred_name])
                if dist > THRESHOLD_DIST:
                    is_unknown = True

            if conf < 0.85 or is_unknown:
                return {"device": "UNKNOWN DEVICE", "confidence": conf, "distance": dist, "status": "CẢNH BÁO: Thiết bị lạ!"}
            return {"device": pred_name, "confidence": conf, "distance": dist, "status": "ĐÃ XÁC MINH"}

if __name__ == "__main__":
    print("\n" + "="*60)
    print("HỆ THỐNG NHẬN DIỆN THIẾT BỊ IOT - SELF-SUPERVISED LEARNING")
    print("="*60)

    BASE_DIR = Path(__file__).resolve().parent
    
    BEST_EXP = BASE_DIR / "experiments" / "finetune_d256"
    if not BEST_EXP.exists():
        BEST_EXP = BASE_DIR.parent / "experiments" / "finetune_d256"
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
            
            device_str = f"{res['device']}"
            conf_str = f"{res['confidence']}"
            dist_str = f"{res['distance']}"
            status_str = f"{res['status']}"

            print(f"KẾT QUẢ NHẬN DIỆN : {device_str}")
            print(f"ĐỘ TIN CẬY        : {conf_str}")
            print(f"KHOẢNG CÁCH       : {dist_str}")
            print(f"TRẠNG THÁI        : {status_str}")

    print("\nCảm ơn bạn đã sử dụng hệ thống!")
