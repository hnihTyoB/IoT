import numpy as np
import json
import yaml
import joblib
from pathlib import Path
import torch
from typing import Any, Optional


def _to_jsonable(obj: Any):
    """Convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj


def save_experiment_artifacts(
    output_dir: str | Path,
    config: dict,
    model: torch.nn.Module,
    metrics: list[dict] | dict,
    feature_scaler=None,
    model_stats: dict | None = None,
    classification_report: str | dict | None = None,
    confusion_matrix_data: str | list | np.ndarray | None = None,
    label_map: dict | None = None,
):
    """Save all experiment artifacts to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(_to_jsonable(config), f, default_flow_style=False)

    # Metrics history
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(_to_jsonable(metrics), f, indent=2)

    # Model weights
    torch.save(model.state_dict(), output_dir / "model.pt")

    # Model stats
    if model_stats is not None:
        stats_save = {
            k: v for k, v in model_stats.items()
            if k not in ("y_true", "y_pred")
        }
        with open(output_dir / "model_stats.json", "w") as f:
            json.dump(_to_jsonable(stats_save), f, indent=2)

    # Classification report
    if classification_report is not None:
        if isinstance(classification_report, dict):
            with open(output_dir / "classification_report.json", "w") as f:
                json.dump(classification_report, f, indent=2)
        else:
            with open(output_dir / "classification_report.txt", "w") as f:
                f.write(str(classification_report))

    # Confusion matrix
    if confusion_matrix_data is not None:
        if isinstance(confusion_matrix_data, np.ndarray):
            np.savetxt(
                output_dir / "confusion_matrix.tsv",
                confusion_matrix_data, fmt="%d", delimiter="\t",
            )
        else:
            with open(output_dir / "confusion_matrix.txt", "w") as f:
                f.write(str(confusion_matrix_data))

    # Label map
    if label_map is not None:
        with open(output_dir / "label_map.json", "w") as f:
            json.dump(_to_jsonable(label_map), f, indent=2)

    # Feature scaler
    if feature_scaler is not None:
        joblib.dump(feature_scaler, output_dir / "scaler_features.pkl")

    print(f"[Saved] All artifacts → {output_dir.resolve()}")
