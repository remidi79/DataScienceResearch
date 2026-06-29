from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def flatten_metrics(path: str | Path, id_fields: list[str] | None = None) -> pd.DataFrame:
    rows = read_jsonl(path)
    id_fields = id_fields or []
    out: list[dict[str, Any]] = []
    for row in rows:
        flat = {field: row.get(field) for field in id_fields}
        identity = row.get("identity") or {}
        for key, value in identity.items():
            flat[f"identity_{key}"] = value
        for key, value in (row.get("metrics") or {}).items():
            flat[key] = value
        out.append(flat)
    return pd.DataFrame(out)

def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
