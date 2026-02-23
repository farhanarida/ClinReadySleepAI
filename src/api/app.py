from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse

from src.api.schemas import (
    InferenceRequest, InferenceResponse,
    BatchInferenceRequest, BatchInferenceResponse, CohortSummary, ExplanationStub
)
from src.api.validators import load_expected_tab_dim, validate_request_shapes, validate_against_limits
from src.api.rate_limit import RateLimiter, RateLimitConfig
from src.export.ort_infer import OrtRunner, OrtConfig
from src.calibration.scaling import OrdinalScaler
from src.pipeline.audit import AuditLogger, AuditConfig
from src.reporting.generate_report import generate_patient_report, ReportConfig
from src.reporting.cohort import cohort_summary_from_rows
from src.utils.config import load_yaml
from src.models.ordinal_numpy import logits_to_probs_numpy
from src.utils.privacy import hash_identifier

app = FastAPI(title="ClinReadySleepAI", version="1.0")

CFG = load_yaml("configs/deploy.yaml")
LIMITS = CFG.get("limits", {})
RL = CFG.get("rate_limit", {})

MODEL_PATH = Path(CFG["paths"]["onnx_model"])
CALIB_PATH = Path(CFG["paths"]["calibration"])
AUDIT_DIR = Path(CFG["paths"]["audit_dir"])

ort = OrtRunner(
    MODEL_PATH,
    OrtConfig(
        intra_op_num_threads=int(CFG["runtime"]["intra_op_num_threads"]),
        inter_op_num_threads=int(CFG["runtime"]["inter_op_num_threads"]),
        enable_mem_pattern=bool(CFG["runtime"]["enable_mem_pattern"]),
        enable_cpu_mem_arena=bool(CFG["runtime"]["enable_cpu_mem_arena"]),
    ),
)
scaler = OrdinalScaler.load(CALIB_PATH)

report_cfg = ReportConfig(
    include_attention=bool(CFG["report"]["include_attention"]),
    include_reliability=bool(CFG["report"]["include_reliability"]),
)

audit = AuditLogger(
    AuditConfig(
        audit_dir=str(AUDIT_DIR),
        model_path=str(MODEL_PATH),
        calib_path=str(CALIB_PATH),
        app_version=str(CFG["app"]["version"]),
    )
)

EXPECTED_TAB_DIM = load_expected_tab_dim("artifacts/preprocess/tabular/tabular_meta.json")

limiter = RateLimiter(
    RateLimitConfig(
        requests_per_minute=int(RL.get("requests_per_minute", 60)),
        burst=int(RL.get("burst", 30)),
    )
)
HASH_SALT = os.getenv("CLINREADY_AUDIT_SALT", "")

def provenance_dict() -> Dict[str, Any]:
    return {
        "app_version": audit.version_stamp.get("app_version"),
        "git_commit": audit.version_stamp.get("git_commit"),
        "git_commit_short": audit.version_stamp.get("git_commit_short"),
        "git_dirty": audit.version_stamp.get("git_dirty"),
        "build_time_utc": audit.version_stamp.get("build_time_utc"),
        "model_sha256": audit.model_hash,
        "calib_sha256": audit.calib_hash,
        "onnx_model_path": str(MODEL_PATH),
        "calibration_path": str(CALIB_PATH),
        "expected_tab_dim": EXPECTED_TAB_DIM,
        "onnx_loaded": bool(MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0),
        "runner_mode": "onnxruntime" if (MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0) else "dummy_fallback",
    }

@app.middleware("http")
async def rate_limit_and_payload_guard(request: Request, call_next):
    max_bytes = int(LIMITS.get("max_payload_bytes", 2_500_000))
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > max_bytes:
                return JSONResponse({"detail": "Payload too large"}, status_code=413)
        except Exception:
            pass

    client_ip = request.client.host if request.client else "unknown"
    if not limiter.allow(client_ip):
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)

    return await call_next(request)

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    return provenance_dict()

def _infer_one(req: InferenceRequest) -> InferenceResponse:
    try:
        validate_request_shapes(
            x_tab_len=len(req.x_tab),
            tab_present=req.tab_present,
            expected_tab_dim=EXPECTED_TAB_DIM,
            sig_len_flat=len(req.x_sig),
            sig_channels=req.sig_channels,
            sig_length=req.sig_length,
            sig_present=req.sig_present,
        )
        validate_against_limits(
            limits=LIMITS,
            x_tab_len=len(req.x_tab),
            sig_channels=req.sig_channels,
            sig_length=req.sig_length,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    x_tab = np.asarray(req.x_tab, dtype=np.float32).reshape(1, -1) if len(req.x_tab) > 0 else np.zeros((1, 0), dtype=np.float32)
    tab_mask = np.array([[1.0 if req.tab_present else 0.0]], dtype=np.float32)
    x_sig = np.asarray(req.x_sig, dtype=np.float32).reshape(1, req.sig_channels, req.sig_length)
    sig_mask = np.array([[1.0 if req.sig_present else 0.0]], dtype=np.float32)

    out = ort.run(x_tab, tab_mask, x_sig, sig_mask)
    logits_ord = np.asarray(out["logits_ord"], dtype=np.float32)
    alpha = np.asarray(out["alpha"], dtype=np.float32)
    r = np.asarray(out["r"], dtype=np.float32)

    # Ordinal-aware scaling (Eq. 12 in paper)
    s = logits_ord.mean(axis=1, keepdims=True)
    s_hat = scaler.a * s + scaler.b
    logits_scaled = logits_ord + (s_hat - s)

    probs = logits_to_probs_numpy(logits_scaled)[0]

    conf = float(np.max(probs))
    pred = int(np.argmax(probs))
    defer = bool(conf < (0.5 + float(CFG["decision"]["deferral_prob_band"])))
    high_risk = bool(probs[-1] >= float(CFG["decision"]["high_risk_prob"]))

    report = generate_patient_report(
        subject_id=req.subject_id,
        probs=probs,
        pred_class=pred,
        conf=conf,
        high_risk=high_risk,
        defer=defer,
        alpha=alpha[0],
        r=r[0],
        cfg=report_cfg,
    )
    report["explanations"] = ExplanationStub(
        available=False,
        method=None,
        summary="Explanation module not enabled in this build; integrate saliency/SHAP outputs here.",
        artifacts={},
    ).model_dump()
    report["provenance"] = provenance_dict()

    # PHI-safe audit (hash subject)
    subject_hash = hash_identifier(req.subject_id, salt=HASH_SALT)
    audit.log({
        "subject_hash": subject_hash,
        "predicted_class": pred,
        "confidence": conf,
        "high_risk": high_risk,
        "defer": defer,
        "probs": report["risk_probabilities"],
    })

    return InferenceResponse(**report)

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest) -> InferenceResponse:
    return _infer_one(req)

@app.post("/infer_batch", response_model=BatchInferenceResponse)
def infer_batch(req: BatchInferenceRequest) -> BatchInferenceResponse:
    max_items = int(LIMITS.get("max_batch_items", 128))
    if len(req.items) > max_items:
        raise HTTPException(status_code=400, detail=f"Batch too large: max_batch_items={max_items}")

    results: List[InferenceResponse] = []
    rows: List[Dict[str, Any]] = []
    for item in req.items:
        r = _infer_one(item)
        results.append(r)
        rows.append(r.model_dump())

    summ = cohort_summary_from_rows(rows)
    cohort = CohortSummary(**summ)

    return BatchInferenceResponse(results=results, cohort_summary=cohort, provenance=provenance_dict())
