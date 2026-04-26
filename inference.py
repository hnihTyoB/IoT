"""
IoT Device Identification — Inference Suite
===========================================
- Load trained models (weights + scaler).
- Identify known IoT devices from CSV traffic.
- Detect "Unknown" devices using confidence thresholding.
"""

import torch
import pandas as pd
import numpy as np
import json
import joblib
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from functions_and_modules.models import IoTDeviceClassifier
from functions_and_modules.dataset import UNSW_NUMERIC_FEATURES

class IoTInferenceEngine:
    def __init__(self, experiment_dir: str, device: str = "cpu"):
        self.exp_dir = Path(experiment_dir)
        self.device = torch.device(device)
        
        # 1. Load configuration and label map
        with open(self.exp_dir / "label_map.json", "r") as f:
            self.label_map = json.load(f)
        self.inv_label_map = {int(v): k for k, v in self.label_map.items()}
        
        with open(self.exp_dir / "config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
            
        # 2. Load feature scaler
        self.scaler = joblib.load(self.exp_dir / "scaler_features.pkl")
        self.feature_names = self.config.get("features", UNSW_NUMERIC_FEATURES)
        
        # 3. Initialize and load model
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
        print(f"🚀 Model loaded successfully from {experiment_dir}")

    def preprocess_flows(self, df: pd.DataFrame) -> torch.Tensor:
        """Preprocess raw flows into scaled window tensors."""
        # Ensure correct features and order
        data = df[self.feature_names].values
        
        # Scale
        data_scaled = self.scaler.transform(data)
        
        # Create windows (using first 10 flows as a window)
        window_size = self.config.get("window_size", 10)
        if len(data_scaled) < window_size:
            # Padding if not enough flows
            pad = np.zeros((window_size - len(data_scaled), data_scaled.shape[1]))
            data_scaled = np.vstack([data_scaled, pad])
            
        window = data_scaled[:window_size] # Take first window
        return torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)

    def predict(self, csv_path: str, confidence_threshold: float = 0.8) -> Dict:
        """Predict the device type from a CSV of flows."""
        df = pd.read_csv(csv_path)
        
        # Handle some missing features if necessary
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
                
        input_tensor = self.preprocess_flows(df)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
            conf = conf.item()
            pred_idx = pred_idx.item()
            
            # unknown detection
            if conf < confidence_threshold:
                result = {
                    "device": "Unknown / Unauthorized Device",
                    "confidence": conf,
                    "status": "⚠️ WARNING: Low confidence detection",
                    "raw_prediction": self.inv_label_map.get(pred_idx, "Unknown")
                }
            else:
                result = {
                    "device": self.inv_label_map.get(pred_idx, "Unknown"),
                    "confidence": conf,
                    "status": "✅ Verified Device"
                }
        
        return result

# --- Example Usage for Demo ---
if __name__ == "__main__":
    # Path to your best experiment
    # Ensure this directory exists on Colab after syncing from Drive
    BEST_EXP = "experiments/finetune_d256"
    
    if not Path(BEST_EXP).exists():
        print(f"❌ Error: Experiment directory {BEST_EXP} not found.")
        print(f"💡 Hint: Run '!cp -r /content/drive/MyDrive/experiments/{Path(BEST_EXP).name} /content/IoT/experiments/' first.")
    else:
        engine = IoTInferenceEngine(BEST_EXP)
        
        # Test with a specific file (replace with any CSV flow file)
        # For demo, you can pick any file from the dataset
        test_file = "dataset/AmazonEcho_44650d56ccd3_flows.csv" 
        
        if Path(test_file).exists():
            print(f"\n🔍 Analyzing traffic from: {test_file}...")
            res = engine.predict(test_file)
            
            print(f"{'='*50}")
            print(f" RESULT     : {res['device']}")
            print(f" CONFIDENCE : {res['confidence']:.2%}")
            print(f" STATUS     : {res['status']}")
            print(f"{'='*50}")
        else:
            print(f"\n💡 Hint: Place a CSV flow file at {test_file} to run a prediction.")
