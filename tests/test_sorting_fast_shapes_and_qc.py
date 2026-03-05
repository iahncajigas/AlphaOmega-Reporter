from __future__ import annotations

import numpy as np

from alphaomega_reporter import ReportConfig
from alphaomega_reporter.sorting_fast import sort_single_channel
from alphaomega_reporter.sorting_spikeinterface import sort_with_spikeinterface
from alphaomega_reporter.synthetic import make_synthetic_session


def test_fast_sorter_returns_waveforms_labels_and_qc() -> None:
    session = make_synthetic_session(depths_mm=[0.0], sides=("L",), random_seed=2)
    segment = session.segments[0]
    values = segment.streams["MER_1"].data[:, 0]

    result = sort_single_channel(values, fs_hz=24000.0, depth_mm=0.0, config=ReportConfig())

    assert result.waveforms.ndim == 2
    assert result.waveforms.shape[0] == result.labels.shape[0] == result.spike_times_s.shape[0]
    assert len(result.unit_summaries) >= 1
    assert all(unit.firing_rate_hz > 0 for unit in result.unit_summaries)
    assert all(np.isfinite(unit.presence_ratio) for unit in result.unit_summaries)


def test_spikeinterface_path_falls_back_cleanly_without_extra() -> None:
    session = make_synthetic_session(depths_mm=[0.0], sides=("L",), random_seed=3)
    segment = session.segments[0]
    values = segment.streams["MER_1"].data[:, 0]

    result = sort_with_spikeinterface(values, fs_hz=24000.0, depth_mm=0.0, config=ReportConfig())

    assert result.spike_times_s.ndim == 1
    assert any("fell back to fast sorter" in warning for warning in result.warnings)
