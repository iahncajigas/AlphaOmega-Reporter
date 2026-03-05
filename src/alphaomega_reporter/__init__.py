"""Standalone Python reporting toolbox for AlphaOmega MER and LFP data."""

from .config import ReportConfig
from .features import (
    apply_notch_filter,
    band_limits,
    compute_bandpower_db,
    compute_multitaper_psd,
    compute_welch_psd,
)
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
    "apply_notch_filter",
    "band_limits",
    "build_report",
    "build_report_from_session",
    "compute_bandpower_db",
    "compute_multitaper_psd",
    "compute_welch_psd",
    "export_h5",
    "load_case",
]

__version__ = "0.1.0"
