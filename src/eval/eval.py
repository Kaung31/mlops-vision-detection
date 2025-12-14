from pathlib import Path
import json
import re
import yaml
import pandas as pd
import mlflow
from ultralytics import YOLO


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_latest_metrics(csv_path: Path):
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    last = df.iloc[-1].to_dict()

    metrics = {}
    for k, v in last.items():
        try:
            metrics[k] = float(v)
        except Exception:
            pass
    return metrics


def main():
    p = load_params()
    t = p["train"]
    paths = p["paths"]
    mcfg = p["mlflow"]

    artifacts_dir = Path(paths["artifacts_dir"])
    outputs_dir = Path(paths["outputs_dir"])
    reports_dir = Path(paths["reports_dir"])

    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model at {model_path}. Run training first.")

    mlflow.set_tracking_uri(mcfg["tracking_uri"])
    mlflow.set_experiment(mcfg["experiment"])

    with mlflow.start_run(run_name="eval"):
        mlflow.log_param("eval_model", str(model_path))
        mlflow.log_param("eval_data", t["data"])

        model = YOLO(str(model_path))

        # Run validation; outputs go to outputs/val/
        model.val(
            data=t["data"],
            imgsz=int(t["imgsz"]),
            conf=float(t["conf"]),
            device=t["device"],
            project=str(outputs_dir),
            name="val",
        )

        val_dir = outputs_dir / "val"
        results_csv = val_dir / "results.csv"
        metrics = read_latest_metrics(results_csv)

        # Log metrics to MLflow
        for k, v in metrics.items():
            mlflow.log_metric(clean_mlflow_key(k), v)

        # Save a clean metrics json for DVC output
        metrics_path = reports_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(metrics_path))

        # Log all validation artifacts
        if val_dir.exists():
            mlflow.log_artifacts(str(val_dir), artifact_path="val_outputs")

        print(f"✅ Eval logged. Metrics saved to {metrics_path}")

def clean_mlflow_key(name: str) -> str:
    # MLflow allows only: alphanumerics, underscores, dashes, periods, spaces, slashes
    return re.sub(r"[^0-9a-zA-Z_\-\. /]", "_", name)


if __name__ == "__main__":
    main()
