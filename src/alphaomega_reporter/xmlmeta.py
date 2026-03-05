from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import ReportConfig
from .model import CaseMetadata

_SIDE_TARGET_RE = re.compile(r"\b(?P<side>[LR])\s*(?P<target>STN|GPI|VIM)\b", re.IGNORECASE)
_BILATERAL_TARGET_RE = re.compile(r"\b(?:BL|B/L)?\s*(?P<target>STN|GPI|VIM)\b", re.IGNORECASE)


def _normalize_target(text: str) -> str:
    target = text.strip().upper()
    if target == "GPI":
        return "GPi"
    return target


def _candidate_xml_paths(case_dir: Path, config: ReportConfig) -> list[Path]:
    primary = sorted(case_dir.glob(config.case_metadata.xml_glob))
    fallback = sorted(case_dir.glob(config.case_metadata.xml_fallback_glob))
    return [*primary, *fallback]


def parse_case_metadata(case_dir: Path, config: ReportConfig) -> CaseMetadata:
    metadata = CaseMetadata(case_name=case_dir.name)
    if not config.case_metadata.xml_enabled:
        return metadata
    candidates = _candidate_xml_paths(case_dir, config)
    if not candidates:
        return metadata
    xml_path = candidates[0]
    metadata.xml_path = str(xml_path)
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:
        metadata.notes.append(f"Failed to parse XML metadata: {exc}")
        return metadata

    texts: list[str] = []
    for tag in ("PhysicianDescription", "PatientNotes"):
        for node in root.iter(tag):
            if node.text and node.text.strip():
                texts.append(node.text.strip())
    metadata.notes.extend(texts)
    joined = " ".join(texts)

    side_matches = list(_SIDE_TARGET_RE.finditer(joined))
    for match in side_matches:
        side = match.group("side").upper()
        metadata.target_by_side[side] = _normalize_target(match.group("target"))

    if not metadata.target_by_side:
        match = _BILATERAL_TARGET_RE.search(joined)
        if match:
            metadata.target_case_default = _normalize_target(match.group("target"))
    return metadata
