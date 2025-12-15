from pathlib import Path
import json
import yaml
import pandas as pd
import re

from ultralytics import YOLO, settings


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_mlflow_key(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_\-\. /]", "_", name)


class Logger:
    def log_param(self, k, v): ...
    def log_metric(self, k, v): ...
    def log_artifact(self, path): ...
    def log_artifacts(self, path, artifact_path=None): ...
    def start_run(self, run_name): return self
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False


class MLflowLogger(Logger):
    def __init__(self, tracking_uri: str, experiment: str):
        import mlflow
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment)
        self._run = None

    def start_run(self, run_name):
        self._run = self.mlflow.start_run(run_name=run_name)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._run is not None:
            self.mlflow.end_run()
        return False

    def log_param(self, k, v):
        self.mlflow.log_param(clean_mlflow_key(str(k)), str(v))

    def log_metric(self, k, v):
        self.mlflow.log_metric(clean_mlflow_key(str(k)), float(v))

    def log_artifact(self, path):
        self.mlflow.log_artifact(str(path))

    def log_artifacts(self, path, artifact_path=None):
        self.mlflow.log_artifacts(str(path), artifact_path=artifact_path)


def make_logger(mcfg: dict) -> Logger:
    enabled = bool(mcfg.get("enabled", False))
    tracking_uri = (mcfg.get("tracking_uri") or "").strip()
    experiment = mcfg.get("experiment", "ppe-detection")
    if not enabled or not tracking_uri:
        return Logger()
    return MLflowLogger(tracking_uri=tracking_uri, experiment=experiment)


def read_latest_metrics(csv_path: Path):
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    last = df.iloc[-1].to_dict()
    out = {}
    for k, v in last.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def main():
    settings.update({"mlflow": False})

    p = load_params()
    t = p["train"]
    paths = p["paths"]
    mcfg = p.get("mlflow", {})

    artifacts_dir = Path(paths["artifacts_dir"])
    outputs_dir = Path(paths["outputs_dir"])
    reports_dir = Path(paths["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model at {model_path}. Run train first.")

    logger = make_logger(mcfg)

    with logger.start_run("eval"):
        logger.log_param("eval_model", str(model_path))
        logger.log_param("eval_data", t["data"])

        model = YOLO(str(model_path))
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

        for k, v in metrics.items():
            logger.log_metric(k, v)

        metrics_path = reports_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.log_artifact(metrics_path)

        if val_dir.exists():
            logger.log_artifacts(val_dir, artifact_path="val_outputs")

        print(f"✅ Eval finished. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
