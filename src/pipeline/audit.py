from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from src.pipeline.version import get_version_stamp

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    path = Path(path)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

@dataclass(frozen=True)
class AuditConfig:
    audit_dir: str
    model_path: str
    calib_path: str
    app_version: str

class AuditLogger:
    def __init__(self, cfg: AuditConfig) -> None:
        self.cfg = cfg
        Path(cfg.audit_dir).mkdir(parents=True, exist_ok=True)
        self.model_hash = sha256_file(cfg.model_path) if Path(cfg.model_path).exists() else "missing"
        self.calib_hash = sha256_file(cfg.calib_path) if Path(cfg.calib_path).exists() else "missing"
        self.version_stamp = get_version_stamp(cfg.app_version)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.log_path = Path(cfg.audit_dir) / f"audit_{day}.jsonl"

    def log(self, record: Dict[str, Any]) -> None:
        base = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_sha256": self.model_hash,
            "calib_sha256": self.calib_hash,
            **self.version_stamp,
        }
        base.update(record)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(base) + "\n")
