from __future__ import annotations
import hashlib

def hash_identifier(value: str, salt: str = "") -> str:
    h = hashlib.sha256()
    h.update((salt + value).encode("utf-8"))
    return h.hexdigest()
