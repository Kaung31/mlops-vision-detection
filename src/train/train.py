from pathlib import Path
import shutil
import re
import yaml
import pandas as pd
import mlflow
from ultralytics import YOLO

import mlflow
from ultralytics import settings

# Disable Ultralytics' built-in MLflow logging (we will log ourselves)
settings.update({"mlflow": False})
mlflow.end_run()  # safety: ensures no run is already open



def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_latest_metrics_from_csv(csv_path: Path):
    if not csv_path.exists():
        print(f"⚠️ results.csv not found at: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    if df.empty:
        return {}

    last = df.iloc[-1].to_dict()

    # Log only numeric metrics
    metrics = {}
    for k, v in last.items():
        try:
            fv = float(v)
            metrics[k] = fv
        except Exception:
            pass

    for k, v in metrics.items():
        mlflow.log_metric(clean_mlflow_key(k), v)

    return metrics


def main():
    p = load_params()
    t = p["train"]
    paths = p["paths"]
    mcfg = p["mlflow"]

    artifacts_dir = Path(paths["artifacts_dir"])
    outputs_dir = Path(paths["outputs_dir"])

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mcfg["tracking_uri"])
    mlflow.set_experiment(mcfg["experiment"])

    with mlflow.start_run(run_name="train"):
        # Log params
        for k, v in t.items():
            mlflow.log_param(k, v)

        # Train
        model = YOLO(t["model"])
        results = model.train(
            data=t["data"],
            epochs=int(t["epochs"]),
            imgsz=int(t["imgsz"]),
            batch=int(t["batch"]),
            device=t["device"],
            project=str(outputs_dir),
            name="train",
        )

        # Ultralytics saves into outputs/train/
        train_dir = outputs_dir / "train"
        weights_dir = train_dir / "weights"
        best_path = weights_dir / "best.pt"
        results_csv = train_dir / "results.csv"

        # Copy best.pt into a stable place for the pipeline
        stable_best = artifacts_dir / "best.pt"
        if best_path.exists():
            shutil.copy2(best_path, stable_best)
            mlflow.log_artifact(str(stable_best))
            print(f"✅ Copied best.pt to {stable_best}")
        else:
            print(f"⚠️ best.pt not found at {best_path}")

        # Log metrics from results.csv (easy + reliable)
        metrics = log_latest_metrics_from_csv(results_csv)

        # Log useful artifacts (plots, csv, etc.)
        if train_dir.exists():
            mlflow.log_artifacts(str(train_dir), artifact_path="train_outputs")

        print("✅ Training run logged to MLflow.")

def clean_mlflow_key(name: str) -> str:
    # MLflow allows only: alphanumerics, underscores, dashes, periods, spaces, slashes
    return re.sub(r"[^0-9a-zA-Z_\-\. /]", "_", name)


if __name__ == "__main__":
    main()
