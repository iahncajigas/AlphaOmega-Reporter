from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from alphaomega_reporter import ReportConfig
from alphaomega_reporter.report import build_report_from_session
from alphaomega_reporter.synthetic import make_synthetic_session


def test_report_build_smoke(tmp_path: Path) -> None:
    session = make_synthetic_session()
    out_path = tmp_path / "report.pdf"

    build_report_from_session(session, out_path, ReportConfig())

    assert out_path.exists()
    assert len(PdfReader(str(out_path)).pages) > 0


def test_report_build_handles_missing_lfp_and_spikes_with_multiband_layout(tmp_path: Path) -> None:
    config = ReportConfig()
    config.render.plot_bands = ["alpha", "gamma"]
    session = make_synthetic_session(
        missing_lfp_depths={0.0},
        missing_spike_depths={-1.0},
    )
    out_path = tmp_path / "report_multiband.pdf"

    build_report_from_session(session, out_path, config)

    assert out_path.exists()
    assert len(PdfReader(str(out_path)).pages) == 7


def test_dense_trajectory_moves_table_to_extra_pages(tmp_path: Path) -> None:
    session = make_synthetic_session(depths_mm=[float(idx) for idx in range(26)], sides=("L",))
    out_path = tmp_path / "dense_report.pdf"

    build_report_from_session(session, out_path, ReportConfig())

    assert out_path.exists()
    assert len(PdfReader(str(out_path)).pages) > 3
