from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

def cohort_summary_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "high_risk_rate": 0.0, "deferral_rate": 0.0, "mean_confidence": 0.0}
    high = np.mean([1.0 if r.get("high_risk_flag") else 0.0 for r in rows])
    defer = np.mean([1.0 if r.get("defer_to_clinician_flag") else 0.0 for r in rows])
    conf = np.mean([float(r.get("confidence", 0.0)) for r in rows])
    return {"n": int(n), "high_risk_rate": float(high), "deferral_rate": float(defer), "mean_confidence": float(conf)}
