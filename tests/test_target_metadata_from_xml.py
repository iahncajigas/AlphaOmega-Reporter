from __future__ import annotations

from alphaomega_reporter.config import ReportConfig, TargetBinding
from alphaomega_reporter.io import resolve_target
from alphaomega_reporter.synthetic import write_stub_case_dir
from alphaomega_reporter.xmlmeta import parse_case_metadata


def test_xml_target_metadata_resolves_bilateral_stn(tmp_path) -> None:
    case_dir = write_stub_case_dir(tmp_path / "case", physician_description="BL STN")

    metadata = parse_case_metadata(case_dir, ReportConfig())

    assert metadata.target_case_default == "STN"
    assert metadata.target_by_side == {}


def test_config_binding_overrides_xml_target(tmp_path) -> None:
    case_dir = write_stub_case_dir(tmp_path / "case", physician_description="BL STN")
    config = ReportConfig()
    config.targets.bindings = [TargetBinding(side="L", trajectory=1, target="GPi")]

    metadata = parse_case_metadata(case_dir, config)

    assert resolve_target("L", 1, metadata, config) == "GPi"
    assert resolve_target("R", 1, metadata, config) == "STN"
