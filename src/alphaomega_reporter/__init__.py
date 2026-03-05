"""Standalone Python reporting toolbox for AlphaOmega MER and LFP data."""

from .config import ReportConfig
from .h5io import export_h5
from .io import load_case
from .model import (
    CaseMetadata,
    ContinuousStream,
    DepthSummaryRow,
    EventSeries,
    Segment,
    Session,
    SpikeStream,
    TrajectoryGroup,
    UnitSummary,
)
from .report import build_report, build_report_from_session

__all__ = [
    "CaseMetadata",
    "ContinuousStream",
    "DepthSummaryRow",
    "EventSeries",
    "ReportConfig",
    "Segment",
    "Session",
    "SpikeStream",
    "TrajectoryGroup",
    "UnitSummary",
    "build_report",
    "build_report_from_session",
    "export_h5",
    "load_case",
]

__version__ = "0.1.0"
