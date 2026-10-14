"""Output conventions: out/<stem>.json is the record, out/<stem>.txt is for reading."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_out(stem: str, payload: dict, text: str) -> tuple[Path, Path]:
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    pj, pt = out / f"{stem}.json", out / f"{stem}.txt"
    pj.write_text(json.dumps(payload, indent=1, default=_json_default))
    pt.write_text(text)
    return pj, pt


def machine_info(extra: dict | None = None) -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["mps"] = bool(torch.backends.mps.is_available())
    except Exception:
        info["torch"] = None
    p = ROOT / "sam2_commit.txt"
    info["sam2_commit"] = p.read_text().strip() if p.exists() else None
    if extra:
        info.update(extra)
    return info
