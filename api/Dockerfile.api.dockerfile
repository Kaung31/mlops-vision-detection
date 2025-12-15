FROM python:3.12-slim

WORKDIR /app

# Avoid python writing pyc and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only API code (keep image smaller)
COPY api /app/api

EXPOSE 8000

# Default: load local model at artifacts/best.pt
# You can also run with:
#   -e MODEL_SOURCE=hf -e HF_REPO=Havertz31/ppe-yolo-v8n-mlops -e HF_FILENAME=best.pt
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
