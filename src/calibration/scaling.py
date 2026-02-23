from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass
class OrdinalScaler:
    a: float = 1.0
    b: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"a": float(self.a), "b": float(self.b)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "OrdinalScaler":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(a=float(d["a"]), b=float(d["b"]))
