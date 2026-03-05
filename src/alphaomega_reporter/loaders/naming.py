from __future__ import annotations

import re
from typing import Any

AO_FILENAME_RE = re.compile(
    r"^(?:(?P<prefix_i>i))?(?:(?P<hemi>[RL]))?T(?P<traj>\d+)"
    r"D(?P<depth>[+-]?\d+(?:\.\d+)?)"
    r"(?P<tag>(?:[+-][^F]+)?)F(?P<file>\d+)$",
    re.IGNORECASE,
)


class FilenameParseError(ValueError):
    pass


def parse_ao_filename(stem: str) -> dict[str, Any]:
    match = AO_FILENAME_RE.match(stem)
    if not match:
        raise FilenameParseError(f"Could not parse AlphaOmega filename stem: '{stem}'")

    hemi = match.group("hemi")
    return {
        "prefix_i": bool(match.group("prefix_i")),
        "hemisphere": hemi.upper() if hemi else None,
        "trajectory_number": int(match.group("traj")),
        "depth_mm": float(match.group("depth")),
        "tag": match.group("tag") or "",
        "file_index": int(match.group("file")),
        "raw_stem": stem,
    }


def sort_key_from_meta(meta: dict[str, Any]) -> tuple:
    hemi_order = {"L": 0, "R": 1, None: 2}
    return (
        hemi_order.get(meta.get("hemisphere"), 3),
        int(meta.get("trajectory_number", 0)),
        float(meta.get("depth_mm", 0.0)),
        str(meta.get("tag", "")),
        int(meta.get("file_index", 0)),
    )
