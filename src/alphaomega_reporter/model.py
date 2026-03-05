from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ContinuousStream:
    stream_id: str
    fs_hz: float
    channel_ids: list[str]
    channel_names: list[str]
    units: str
    data: np.ndarray
    t0_s: float = 0.0
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpikeStream:
    spike_id: str
    fs_hz: float
    times_s: np.ndarray
    unit_ids: np.ndarray | None = None
    waveforms: np.ndarray | None = None
    pre_s: float | None = None
    post_s: float | None = None
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventSeries:
    event_id: str
    times_s: np.ndarray
    values: np.ndarray | list[str] | None = None
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class Segment:
    segment_id: str
    meta: dict[str, Any] = field(default_factory=dict)
    streams: dict[str, ContinuousStream] = field(default_factory=dict)
    spikes: dict[str, SpikeStream] = field(default_factory=dict)
    events: dict[str, EventSeries] = field(default_factory=dict)


@dataclass
class Session:
    session_path: str
    source_files: list[str] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseMetadata:
    case_name: str
    xml_path: str | None = None
    target_case_default: str | None = None
    target_by_side: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class SelectedStream:
    stream_key: str
    stream_id: str
    channel_index: int
    channel_id: str
    channel_name: str
    fs_hz: float
    units: str


@dataclass
class UnitSummary:
    depth_mm: float
    unit_id: int
    spike_times_s: np.ndarray
    firing_rate_hz: float
    median_p2p: float
    isi_violation_ratio: float
    presence_ratio: float
    amplitude_cutoff_proxy: float


@dataclass
class DepthSummaryRow:
    depth_mm: float
    primary_band_power_db: float | None
    spike_count: int
    n_units: int
    median_fr_hz: float | None
    median_amp_p2p: float | None
    median_isi_violation_ratio: float | None
    median_presence_ratio: float | None
    median_amplitude_cutoff_proxy: float | None
    mer_rms: float | None
    lfp_available: bool
    mer_available: bool
    warning_flags: list[str] = field(default_factory=list)
    extra_band_powers_db: dict[str, float] = field(default_factory=dict)


@dataclass
class TrajectoryGroup:
    side: str
    trajectory: int
    target: str | None
    segments: list[Segment]
    depth_values: list[float]
    warnings: list[str] = field(default_factory=list)
