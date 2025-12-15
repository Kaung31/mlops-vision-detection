from pathlib import Path
import shutil
import yaml
import pandas as pd
import re

from ultralytics import YOLO, settings


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_mlflow_key(name: str) -> str:
    # MLflow metric/param names: alphanumerics, underscores, dashes, periods, spaces, slashes
    return re.sub(r"[^0-9a-zA-Z_\-\. /]", "_", name)


class Logger:
    """No-op logger by default; becomes MLflow logger when enabled."""
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

    def __enter__(self):  # for "with logger.start_run(...)"
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

    if not enabled:
        return Logger()

    if not tracking_uri:
        # enabled=True but no URI → treat as disabled to avoid crashes
        return Logger()

    return MLflowLogger(tracking_uri=tracking_uri, experiment=experiment)


def log_latest_metrics_from_csv(logger: Logger, csv_path: Path):
    if not csv_path.exists():
        print(f"⚠️ results.csv not found at: {csv_path}")
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

    for k, v in metrics.items():
        logger.log_metric(k, v)

    return metrics


def main():
    # Disable Ultralytics' built-in MLflow integration to avoid conflicts
    settings.update({"mlflow": False})

    p = load_params()
    t = p["train"]
    paths = p["paths"]
    mcfg = p.get("mlflow", {})

    artifacts_dir = Path(paths["artifacts_dir"])
    outputs_dir = Path(paths["outputs_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger = make_logger(mcfg)

    with logger.start_run("train"):
        for k, v in t.items():
            logger.log_param(k, v)

        model = YOLO(t["model"])
        model.train(
            data=t["data"],
            epochs=int(t["epochs"]),
            imgsz=int(t["imgsz"]),
            batch=int(t["batch"]),
            device=t["device"],
            project=str(outputs_dir),
            name="train",
        )

        train_dir = outputs_dir / "train"
        weights_dir = train_dir / "weights"
        best_path = weights_dir / "best.pt"
        results_csv = train_dir / "results.csv"

        stable_best = artifacts_dir / "best.pt"
        if best_path.exists():
            shutil.copy2(best_path, stable_best)
            logger.log_artifact(stable_best)
            print(f"✅ Copied best.pt to {stable_best}")
        else:
            print(f"⚠️ best.pt not found at {best_path}")

        log_latest_metrics_from_csv(logger, results_csv)

        if train_dir.exists():
            logger.log_artifacts(train_dir, artifact_path="train_outputs")

        print("✅ Training stage finished.")


if __name__ == "__main__":
    main()
