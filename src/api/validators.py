from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

def load_expected_tab_dim(tab_meta_path: str | Path = "artifacts/preprocess/tabular/tabular_meta.json") -> int:
    d = json.loads(Path(tab_meta_path).read_text(encoding="utf-8"))
    feats = d.get("feature_names", [])
    return int(len(feats))

def validate_request_shapes(
    x_tab_len: int,
    tab_present: bool,
    expected_tab_dim: int,
    sig_len_flat: int,
    sig_channels: int,
    sig_length: int,
    sig_present: bool,
) -> None:
    if tab_present and expected_tab_dim > 0 and x_tab_len != expected_tab_dim:
        raise ValueError(f"x_tab length mismatch: got {x_tab_len}, expected {expected_tab_dim}")
    if sig_present:
        if sig_channels <= 0 or sig_length <= 0:
            raise ValueError("sig_channels and sig_length must be positive when sig_present=True")
        expected_flat = sig_channels * sig_length
        if sig_len_flat != expected_flat:
            raise ValueError(f"x_sig length mismatch: got {sig_len_flat}, expected {expected_flat} (C*T)")

def validate_against_limits(
    *,
    limits: Dict[str, Any],
    x_tab_len: int,
    sig_channels: int,
    sig_length: int,
) -> None:
    if x_tab_len > int(limits.get("max_tab_dim", 2000)):
        raise ValueError(f"x_tab length exceeds max_tab_dim={limits.get('max_tab_dim')}")
    if sig_channels > int(limits.get("max_sig_channels", 16)):
        raise ValueError(f"sig_channels exceeds max_sig_channels={limits.get('max_sig_channels')}")
    if sig_length > int(limits.get("max_sig_length", 180000)):
        raise ValueError(f"sig_length exceeds max_sig_length={limits.get('max_sig_length')}")
