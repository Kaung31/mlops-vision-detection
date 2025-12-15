import io
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

# Optional: allow downloading model from Hugging Face at startup
USE_HF = os.getenv("MODEL_SOURCE", "local").lower() == "hf"
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/best.pt")
HF_REPO = os.getenv("HF_REPO", "Havertz31/ppe-yolo-v8n-mlops")
HF_FILENAME = os.getenv("HF_FILENAME", "best.pt")
CONF_DEFAULT = float(os.getenv("CONF_DEFAULT", "0.25"))
IOU_DEFAULT = float(os.getenv("IOU_DEFAULT", "0.6"))

# For CI unit tests (don’t load huge model)
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "false").lower() in ("1", "true", "yes")

app = FastAPI(
    title="PPE Detection API",
    version="1.0.0",
    description="YOLOv8 PPE detection inference API (FastAPI)."
)

model = None
names: Dict[int, str] = {}


def _load_model():
    global model, names

    if SKIP_MODEL_LOAD:
        return

    if USE_HF:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
        path = local_path
    else:
        path = MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. "
            f"Set MODEL_PATH or use MODEL_SOURCE=hf with HF_REPO/HF_FILENAME."
        )

    from ultralytics import YOLO
    model = YOLO(path)
    # names mapping is stored in result objects too, but we keep a fallback
    try:
        # Not always available directly, so keep empty fallback
        names = getattr(model, "names", {}) or {}
    except Exception:
        names = {}


@app.on_event("startup")
def startup_event():
    try:
        _load_model()
    except Exception as e:
        # API stays up but /predict returns 503 with the reason
        app.state.model_load_error = str(e)
    else:
        app.state.model_load_error = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_source": "hf" if USE_HF else "local",
        "model_path": (f"{HF_REPO}/{HF_FILENAME}" if USE_HF else MODEL_PATH),
        "error": app.state.model_load_error,
    }


def _ensure_model():
    if model is None:
        msg = app.state.model_load_error or "Model not loaded."
        raise HTTPException(status_code=503, detail=msg)


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")


def _to_detections(results) -> List[Dict[str, Any]]:
    r = results[0]
    dets = []

    if r.boxes is None or len(r.boxes) == 0:
        return dets

    # Prefer r.names (most reliable)
    r_names = getattr(r, "names", None) or names or {}

    for b in r.boxes:
        cls_id = int(b.cls.item())
        conf = float(b.conf.item()) if b.conf is not None else None
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        dets.append({
            "class_id": cls_id,
            "class_name": r_names.get(cls_id, str(cls_id)),
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2],
        })
    return dets


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = CONF_DEFAULT,
    iou: float = IOU_DEFAULT,
    return_annotated: bool = False,
):
    """
    Upload an image and get detections JSON.
    If return_annotated=true, returns base64-unsafe binary PNG via /predict-image instead.
    """
    _ensure_model()

    file_bytes = await file.read()
    img = _read_image(file_bytes)

    np_img = np.array(img)  # RGB numpy
    results = model.predict(source=np_img, conf=conf, iou=iou, verbose=False)

    dets = _to_detections(results)

    payload = {
        "filename": file.filename,
        "conf": conf,
        "iou": iou,
        "num_detections": len(dets),
        "detections": dets,
    }

    if return_annotated:
        payload["note"] = "Use POST /predict-image to get the annotated image bytes."

    return JSONResponse(payload)


@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = CONF_DEFAULT,
    iou: float = IOU_DEFAULT,
):
    """
    Upload an image and get an annotated PNG image back.
    """
    _ensure_model()

    file_bytes = await file.read()
    img = _read_image(file_bytes)

    np_img = np.array(img)  # RGB numpy
    results = model.predict(source=np_img, conf=conf, iou=iou, verbose=False)

    plotted_bgr = results[0].plot()
    plotted_rgb = plotted_bgr[:, :, ::-1]

    out = Image.fromarray(plotted_rgb)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
