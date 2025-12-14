import mlflow
import os

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("ppe-detection")

with mlflow.start_run():
    mlflow.log_param("stage", "docker-mlflow-check")
    mlflow.log_param("model_path", "models/best.pt")
    mlflow.log_metric("sanity", 1.0)
    mlflow.log_artifact("models/best.pt")

print("✅ Logged run to MLflow")
