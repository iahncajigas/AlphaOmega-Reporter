from __future__ import annotations

from pathlib import Path


def read_list_file(path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = raw_line.strip().strip('"')
        if not text:
            continue
        lines.append(text)
    return lines


def resolve_list_entries(base_dir: Path, entries: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for entry in entries:
        item = Path(entry)
        if not item.is_absolute():
            item = (base_dir / entry).resolve()
        if item.exists():
            resolved.append(item)
    return resolved
