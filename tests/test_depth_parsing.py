from __future__ import annotations

import pytest

from alphaomega_reporter.config import ReportConfig
from alphaomega_reporter.io import AlphaOmegaReporterError, normalize_segment_metadata
from alphaomega_reporter.model import Segment, Session


def test_depth_parsing_prefers_meta_depth() -> None:
    session = Session(
        session_path="/tmp/case",
        segments=[
            Segment(
                segment_id="RT1D-2.50",
                meta={"depth_mm": 1.25, "source_file": "/tmp/RT1D-2.50.map"},
            )
        ],
    )

    normalize_segment_metadata(session, ReportConfig())

    segment = session.segments[0]
    assert segment.meta["depth_mm"] == pytest.approx(1.25)
    assert segment.meta["hemisphere"] == "R"
    assert segment.meta["trajectory_number"] == 1


def test_depth_parsing_falls_back_to_source_file() -> None:
    session = Session(
        session_path="/tmp/case",
        segments=[
            Segment(
                segment_id="segment_without_depth",
                meta={"source_file": "/tmp/LT2D+3.50.map"},
            )
        ],
    )

    normalize_segment_metadata(session, ReportConfig())

    segment = session.segments[0]
    assert segment.meta["depth_mm"] == pytest.approx(3.5)
    assert segment.meta["hemisphere"] == "L"
    assert segment.meta["trajectory_number"] == 2


def test_depth_parsing_raises_when_missing_everywhere() -> None:
    session = Session(
        session_path="/tmp/case",
        segments=[Segment(segment_id="segment", meta={"source_file": "segment.map"})],
    )

    with pytest.raises(AlphaOmegaReporterError, match="no segments with depth metadata found"):
        normalize_segment_metadata(session, ReportConfig())
