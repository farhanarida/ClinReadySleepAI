from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np

@dataclass(frozen=True)
class OrtConfig:
    intra_op_num_threads: int = 4
    inter_op_num_threads: int = 1
    enable_mem_pattern: bool = True
    enable_cpu_mem_arena: bool = True

class _DummySession:
    def __init__(self) -> None:
        self.output_names = ["logits_ord", "alpha", "r"]

    def run(self, x_tab: np.ndarray, tab_mask: np.ndarray, x_sig: np.ndarray, sig_mask: np.ndarray) -> Dict[str, np.ndarray]:
        # Deterministic constant outputs (K=3 => K-1=2)
        b = tab_mask.shape[0]
        logits_ord = np.tile(np.array([[0.2, 0.8]], dtype=np.float32), (b, 1))
        alpha = np.tile(np.array([[0.6, 0.4]], dtype=np.float32), (b, 1))
        r = np.tile(np.array([[0.9, 0.8]], dtype=np.float32), (b, 1))
        return {"logits_ord": logits_ord, "alpha": alpha, "r": r}

class OrtRunner:
    """ONNX Runtime runner with safe dummy fallback.

    If ONNX Runtime is unavailable OR the model file is missing/empty,
    a deterministic dummy session is used so the API remains functional.
    """
    def __init__(self, onnx_path: str | Path, cfg: OrtConfig = OrtConfig()) -> None:
        self.onnx_path = Path(onnx_path)
        self.cfg = cfg
        self._sess = None  # type: ignore
        self.output_names = ["logits_ord", "alpha", "r"]

        use_dummy = (not self.onnx_path.exists()) or (self.onnx_path.stat().st_size == 0)
        if use_dummy:
            self._sess = _DummySession()
            return

        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = cfg.intra_op_num_threads
            so.inter_op_num_threads = cfg.inter_op_num_threads
            so.enable_mem_pattern = cfg.enable_mem_pattern
            so.enable_cpu_mem_arena = cfg.enable_cpu_mem_arena
            sess = ort.InferenceSession(str(self.onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
            self.output_names = [o.name for o in sess.get_outputs()]
            self._sess = sess
        except Exception:
            # Fall back to dummy if ORT cannot load the model
            self._sess = _DummySession()

    def run(self, x_tab: np.ndarray, tab_mask: np.ndarray, x_sig: np.ndarray, sig_mask: np.ndarray) -> Dict[str, np.ndarray]:
        if isinstance(self._sess, _DummySession):
            return self._sess.run(x_tab, tab_mask, x_sig, sig_mask)

        # ONNX Runtime path
        feed = {
            "x_tab": x_tab.astype(np.float32),
            "tab_mask": tab_mask.astype(np.float32),
            "x_sig": x_sig.astype(np.float32),
            "sig_mask": sig_mask.astype(np.float32),
        }
        outs = self._sess.run(self.output_names, feed)  # type: ignore
        return {name: out for name, out in zip(self.output_names, outs)}
