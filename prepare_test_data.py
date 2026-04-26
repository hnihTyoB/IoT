"""
Prepare Clean Test Data
=======================
Extracts the 20% validation/test split from the original dataset CSVs
that the model NEVER saw during training. Use these for a high-quality demo.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

def prepare_clean_test_files():
    # 1. Create output directory
    output_dir = Path("demo_data/known")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Find all dataset files
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        print(f"❌ Error: Dataset directory '{dataset_dir}' not found.")
        return

    files = list(dataset_dir.glob("*.csv"))
    if not files:
        print("❌ Error: No CSV files found in dataset directory.")
        return

    print(f"📦 Found {len(files)} devices. Extracting clean test samples...")

    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) < 50:
                print(f" ⚠️ Skipping {f.name}: Too few rows ({len(df)})")
                continue
            
            # Use EXACT SAME SPLIT as in run_experiments.py (test_size=0.2, random_state=42)
            # This ensures we get the exact rows the model NEVER trained on.
            _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=None)
            
            # Take a manageable sample for demo (e.g., up to 1000 rows)
            sample_size = min(len(test_df), 1000)
            demo_sample = test_df.sample(n=sample_size, random_state=7)
            
            # Save to demo_data/known/
            device_name = f.stem.split('_')[0]
            save_path = output_dir / f"test_{device_name}.csv"
            demo_sample.to_csv(save_path, index=False)
            print(f" ✅ Created: {save_path.name} ({sample_size} clean flows)")
            
        except Exception as e:
            print(f" ❌ Failed to process {f.name}: {e}")

    print("\n" + "="*50)
    print("🚀 SUCCESS!")
    print(f"Clean test files are ready in: {output_dir.resolve()}")
    print("Use these files with 'inference.py' for an honest demo.")
    print("="*50)

if __name__ == "__main__":
    prepare_clean_test_files()
