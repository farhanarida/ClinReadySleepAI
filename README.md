
# ClinReadySleepAI
A Distilled, Drift-Aware, and Quantized Multi-Input Deep Learning Framework for Deployable Sleep Apnea Risk Prediction

## Overview
ClinReadySleepAI is a clinically deployable AI inference system designed for real-world sleep apnea risk prediction using multi-input deep learning. 
The framework integrates distillation, drift-aware calibration, quantized CPU inference, and a containerized clinician pipeline.

Key capabilities:
- Multi-input risk prediction (clinical, demographic, genomic, physiological signals)
- Reliability-aware fusion outputs
- Deployment-oriented evaluation
- PHI-safe audit logging
- Docker-based clinical deployment

---

## Repository Structure
ClinReadySleepAI/

├── configs/

│   └── deploy.yaml

├── src/

│     ├── api/

│     ├── calibration/

│     ├── export/

│     ├── reporting/

│     ├── pipeline/

│     ├── models/

│     └── utils/


├── artifacts/

│     ├── onnx/

│     ├── calibration/

│     └── preprocess/


├── docker/


└── scripts/

---

## Installation

Clone repository:

git clone https://github.com/YOUR_USERNAME/ClinReadySleepAI.git
cd ClinReadySleepAI

Create environment:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---

## Running the API

python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Open API docs:
http://localhost:8000/docs

---

## Docker Deployment

docker build -f docker/Dockerfile -t clinreadysleepai:latest .
docker run --rm -p 8000:8000 -e CLINREADY_AUDIT_SALT="change_me" clinreadysleepai:latest

---

## API Endpoints

GET /health
GET /metadata
POST /infer
POST /infer_batch

---

## Privacy and Governance

- Only hashed subject identifiers stored in audit logs
- No raw patient data saved
- Versioned model and calibration hashes included in responses

Audit logs:
artifacts/audit/

---

## Model Artifacts

Place exported ONNX student model at:
artifacts/onnx/student_int8.onnx

Place calibration parameters at:
artifacts/calibration/calib_params.json

---

## Evaluation Philosophy

- Temporal validation (SHHS)
- Frozen external validation (MESA)
- Drift-aware calibration
- CPU latency profiling

---

## Citation

ClinReadySleepAI: A Distilled, Drift-Aware, and Quantized Multi-Input Deep Learning Framework for Deployable Sleep Apnea Risk Prediction.

---

## License

MIT License
