# 🦺 PPE Detection (YOLOv8) — End-to-End MLOps Vision Pipeline

A computer-vision project that detects PPE and missing PPE using a YOLOv8 model, with a reproducible training/evaluation pipeline using **DVC** and experiment tracking with **MLflow**.

## 🔗 Links
- **Live demo (Hugging Face Space):** https://huggingface.co/spaces/Havertz31/ppe-detector-demo  
- **GitHub repo:** https://github.com/Kaung31/mlops-vision-detection  

---

## ✅ What this project does
- Detects PPE + missing PPE classes from images/webcam frames
- Runs a repeatable pipeline:
  - **Train → Evaluate → Save artifacts → Track results**
- Supports:
  - **Local CPU testing** (quick pipeline check)
  - **Google Colab GPU training** (for best performance)
- Deploys the model to:
  - **Hugging Face Model repo**
  - **Hugging Face Space (Gradio UI)**

---

## 🧰 Tech stack
- **Model:** Ultralytics YOLOv8 (object detection)
- **Pipeline:** DVC (`dvc.yaml`, `dvc.lock`)
- **Tracking:** MLflow (local, via Docker Compose)
- **Deployment:** Hugging Face Model repo + Hugging Face Space (Gradio)

---

## 🗂️ Project structure

```text
mlops-vision-detection/
├── .dvc/
├── datasets/
│   └── construction-ppe.yaml
├── models/
│   ├── .gitignore
│   └── best.pt.dvc
├── samples/
├── src/
│   ├── eval/
│   │   └── eval.py
│   └── train/
│       └── train.py
├── .dvcignore
├── .gitignore
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── params.colab.yaml
└── README.md

```
---

## 🚀 Quickstart (Windows — local CPU test)

### 1) Create & activate a virtual environment

- python -m venv .venv
- .\.venv\Scripts\Activate.ps1
- python -m pip install -U pip

### 2) Install dependencies

- pip install -r requirements.txt
- pip install dvc mlflow ultralytics pyyaml pandas

### 3) Start MLflow (optional but recommended)

- docker compose up -d
- Open MLflow UI: http://127.0.0.1:5000

### 4) Run the pipeline
- If dvc is not recognized on Windows, run it via Python:
- python -m dvc repro
- This will:
  - train (CPU test run)
  - evaluate
  -  save reports/metrics.json
  -  produce artifacts/best.pt
  -  log runs to MLflow (if enabled locally)

## ⚡ Train on Google Colab (GPU) — recommended for best model

Local CPU is mainly for testing the pipeline. For a better model, train on Colab GPU:

#### 1.Clone the repo in Colab
#### 2.Install dependencies
#### 3.Switch to GPU params and run the pipeline:

- cp params.colab.yaml params.yaml
- python -m dvc repro


## Outputs:

- artifacts/best.pt
- eports/metrics.json

✅ Save them to Google Drive so you don’t lose them when the Colab session resets.

##  🌍 Deployment (Hugging Face)

The demo Space loads best.pt from the Hugging Face model repo.

### Updating the live demo model

- Retrain on Colab (GPU)

- Upload the new best.pt to your Hugging Face model repo

- The Space automatically uses the latest best.pt

##  🧠 Why this counts as “MLOps”

- DVC: reproducible pipelines (train/eval stages, tracked outputs)

- MLflow: experiment tracking (params, metrics, artifacts)

- Hugging Face Space: public deployment + demo UI

## 📌 Demo

Live Space: https://huggingface.co/spaces/Havertz31/ppe-detector-demo

## 📄 License

This project is released under the MIT License.
