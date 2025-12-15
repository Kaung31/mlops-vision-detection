import os

os.environ["SKIP_MODEL_LOAD"] = "true"

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    # model should be skipped in CI
    assert data["model_loaded"] is False

def test_predict_returns_503_when_model_not_loaded():
    r = client.post("/predict")
    assert r.status_code in (422, 503)
