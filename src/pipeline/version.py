from __future__ import annotations
import subprocess
from datetime import datetime, timezone
from typing import Dict, Optional

def _run_git(args) -> Optional[str]:
    try:
        out = subprocess.check_output(["git"] + args, stderr=subprocess.DEVNULL).decode().strip()
        return out or None
    except Exception:
        return None

def get_version_stamp(app_version: str) -> Dict[str, str]:
    commit = _run_git(["rev-parse", "HEAD"])
    short = _run_git(["rev-parse", "--short", "HEAD"])
    dirty = _run_git(["status", "--porcelain"])
    is_dirty = "true" if (dirty is not None and dirty != "") else "false"
    return {
        "app_version": app_version,
        "git_commit": commit or "unknown",
        "git_commit_short": short or "unknown",
        "git_dirty": is_dirty,
        "build_time_utc": datetime.now(timezone.utc).isoformat(),
    }
