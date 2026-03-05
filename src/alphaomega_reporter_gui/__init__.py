from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_ROOT / "resources"

__all__ = ["PACKAGE_ROOT", "RESOURCE_DIR"]
