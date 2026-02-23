from __future__ import annotations
import numpy as np

def logits_to_probs_numpy(logits_ord: np.ndarray) -> np.ndarray:
    logits_ord = np.asarray(logits_ord, dtype=np.float32)
    s = 1.0 / (1.0 + np.exp(-logits_ord))
    b, km1 = s.shape
    K = km1 + 1
    probs = np.zeros((b, K), dtype=np.float32)
    probs[:, 0] = 1.0 - s[:, 0]
    for k in range(1, km1):
        probs[:, k] = s[:, k - 1] - s[:, k]
    probs[:, K - 1] = s[:, km1 - 1]
    probs = np.clip(probs, 1e-7, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs
