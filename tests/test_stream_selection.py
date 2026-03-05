from __future__ import annotations

import numpy as np

from alphaomega_reporter.config import ReportConfig
from alphaomega_reporter.io import select_lfp_stream, select_mer_stream
from alphaomega_reporter.model import ContinuousStream, Segment


def test_mer_stream_selection_prefers_hint_and_highest_rms_channel() -> None:
    stream = ContinuousStream(
        stream_id="RAW",
        fs_hz=24000.0,
        channel_ids=["0", "1"],
        channel_names=["RAW 0", "RAW 1"],
        units="uV",
        data=np.column_stack(
            [
                np.ones(2000, dtype=np.float32),
                np.linspace(-20.0, 20.0, 2000, dtype=np.float32),
            ]
        ),
    )
    segment = Segment(
        segment_id="LT1D+0.0",
        meta={"depth_mm": 0.0},
        streams={
            "RAW_1": stream,
            "AUX_1": ContinuousStream(
                stream_id="AUX",
                fs_hz=30000.0,
                channel_ids=["0"],
                channel_names=["AUX 0"],
                units="uV",
                data=np.zeros((2000, 1), dtype=np.float32),
            ),
        },
    )

    selected = select_mer_stream(segment, ReportConfig())

    assert selected is not None
    assert selected.stream_key == "RAW_1"
    assert selected.channel_index == 1


def test_lfp_stream_selection_falls_back_to_lowest_fs_without_hints() -> None:
    config = ReportConfig()
    config.stream_selection.lfp_name_hints = []
    segment = Segment(
        segment_id="LT1D+0.0",
        meta={"depth_mm": 0.0},
        streams={
            "A_1": ContinuousStream(
                stream_id="A",
                fs_hz=2000.0,
                channel_ids=["0"],
                channel_names=["A 0"],
                units="uV",
                data=np.zeros((2000, 1), dtype=np.float32),
            ),
            "B_1": ContinuousStream(
                stream_id="B",
                fs_hz=500.0,
                channel_ids=["0"],
                channel_names=["B 0"],
                units="uV",
                data=np.zeros((2000, 1), dtype=np.float32),
            ),
        },
    )

    selected = select_lfp_stream(segment, config)

    assert selected is not None
    assert selected.stream_key == "B_1"
    assert selected.fs_hz == 500.0
