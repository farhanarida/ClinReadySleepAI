# ClinReadySleepAI (Objective-3 Deployable Codebase)

This is a **fully functional** clinician-facing inference service with:
- ONNX Runtime CPU inference (if `artifacts/onnx/student_int8.onnx` is provided)
- Safe **dummy fallback** runner if the ONNX model is missing (so API still runs)
- Ordinal-aware calibration (`a,b` scaling)
- Rate limiting + request-size caps
- PHI-safe audit logging (hashed subject_id)
- Single and batch inference endpoints
- Docker packaging

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Health:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata
```

Demo inference (tab dim = 3):
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "P001",
    "x_tab": [0.1, -0.2, 0.05],
    "tab_present": true,
    "x_sig": [0,0,0,0],
    "sig_channels": 2,
    "sig_length": 2,
    "sig_present": true
  }'
```

## Docker
```bash
docker build -f docker/Dockerfile -t clinreadysleepai:latest .
docker run --rm -p 8000:8000 -e CLINREADY_AUDIT_SALT="change_me" clinreadysleepai:latest
```

Audit logs (JSONL):
- `artifacts/audit/audit_YYYY-MM-DD.jsonl`

> Replace the dummy fallback by exporting your trained student to ONNX at `artifacts/onnx/student_int8.onnx`.
