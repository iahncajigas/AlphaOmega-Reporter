from __future__ import annotations

from alphaomega_reporter import ReportConfig
from alphaomega_reporter.io import _detect_case_kind, load_case
from alphaomega_reporter.loaders.map_native import MAPSessionReader
from alphaomega_reporter.synthetic import make_synthetic_session, write_stub_case_dir


def test_detect_case_kind_prefers_root_map_files(tmp_path) -> None:
    case_dir = write_stub_case_dir(tmp_path / "case")

    kind, files = _detect_case_kind(case_dir, ReportConfig())

    assert kind == "map"
    assert files
    assert all(path.suffix.lower() == ".map" for path in files)


def test_load_case_ignores_non_input_artifacts(tmp_path, monkeypatch) -> None:
    case_dir = write_stub_case_dir(tmp_path / "case")
    synthetic = make_synthetic_session(case_name=case_dir.name)

    def fake_read(self, scale_to_uv=False, concat_depth_files=True):
        return synthetic

    monkeypatch.setattr(MAPSessionReader, "read", fake_read)
    session = load_case(case_dir, ReportConfig())

    assert session.meta["case_name"] == case_dir.name
    assert session.meta["case_metadata"]["target_case_default"] == "STN"
