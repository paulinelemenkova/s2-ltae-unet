from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


def load(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    p = Path(path or os.environ.get("RS_CONFIG", "config.json"))
    if not p.exists():
        raise SystemExit(f"configuration file not found: {p} (see config.example.json)")
    with open(p) as fh:
        cfg: dict[str, Any] = json.load(fh)
    return cfg


def burn_date(cfg: dict[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(cfg["burn_date"])


def horizon(cfg: dict[str, Any]) -> pd.PeriodIndex:
    return pd.period_range(cfg["horizon"][0], cfg["horizon"][1], freq="M")


def glo30_tiles(bbox: list[float]) -> list[str]:
    w, s, e, n = bbox
    tiles = []
    for lat in range(math.floor(s), math.floor(n) + 1):
        for lon in range(math.floor(w), math.floor(e) + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tiles.append(f"{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00")
    return tiles


def gfc_tile(bbox: list[float]) -> str:
    w, s, e, n = bbox
    top = math.ceil(n / 10) * 10
    left = math.floor(w / 10) * 10
    if math.ceil(s / 10) * 10 != top or math.floor(e / 10) * 10 != left:
        raise SystemExit("area of interest spans more than one 10-degree GFC tile")
    ns = "N" if top >= 0 else "S"
    ew = "E" if left >= 0 else "W"
    return f"{abs(top):02d}{ns}_{abs(left):03d}{ew}"
