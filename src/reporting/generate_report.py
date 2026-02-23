from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass(frozen=True)
class ReportConfig:
    include_attention: bool = True
    include_reliability: bool = True

def _tolist(x: np.ndarray) -> List[float]:
    return [float(v) for v in x.tolist()]

def generate_patient_report(
    subject_id: str,
    probs: np.ndarray,
    pred_class: int,
    conf: float,
    high_risk: bool,
    defer: bool,
    alpha: Optional[np.ndarray] = None,
    r: Optional[np.ndarray] = None,
    cfg: ReportConfig = ReportConfig(),
) -> Dict[str, Any]:
    report = {
        "subject_id": subject_id,
        "risk_probabilities": _tolist(probs),
        "predicted_risk_class": int(pred_class),
        "confidence": float(conf),
        "high_risk_flag": bool(high_risk),
        "defer_to_clinician_flag": bool(defer),
        "interpretation_note": "Calibrated risk probability; defer flag indicates low confidence requiring clinician review.",
    }
    if cfg.include_attention and alpha is not None:
        report["modality_attention"] = _tolist(alpha)
    if cfg.include_reliability and r is not None:
        report["modality_reliability"] = _tolist(r)
    return report
