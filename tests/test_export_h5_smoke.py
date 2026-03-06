from __future__ import annotations

import h5py

from alphaomega_reporter import ReportConfig
from alphaomega_reporter.h5io import export_h5
from alphaomega_reporter.synthetic import make_synthetic_session, write_stub_case_dir


def test_export_h5_smoke(tmp_path, monkeypatch) -> None:
    case_dir = write_stub_case_dir(tmp_path / "case")
    session = make_synthetic_session(case_name=case_dir.name)

    monkeypatch.setattr("alphaomega_reporter.h5io.load_case", lambda case_dir_arg, config: session)
    out_path = export_h5(case_dir, tmp_path / "session.h5", ReportConfig())

    with h5py.File(out_path, "r") as h5:
        assert "meta" in h5
        assert "segments" in h5
        assert len(h5["segments"]) == len(session.segments)
