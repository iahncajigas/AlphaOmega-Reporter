from __future__ import annotations

from pathlib import Path

from alphaomega_reporter_gui.report_config import GuiReportConfig


def test_gui_config_roundtrip_yaml_and_json(tmp_path: Path) -> None:
    config = GuiReportConfig()
    config.report.render.include_mua_rms_panel = True
    config.report.render.include_summary_table = False
    config.lfp_view.band_preset = "custom"
    config.lfp_view.custom_band_low_hz = 18.0
    config.lfp_view.custom_band_high_hz = 26.0
    config.mer.notch_hz = 60.0

    yaml_path = tmp_path / "gui_config.yaml"
    json_path = tmp_path / "gui_config.json"
    config.save(yaml_path)
    config.save(json_path)

    yaml_loaded = GuiReportConfig.from_path(yaml_path)
    json_loaded = GuiReportConfig.from_path(json_path)

    assert yaml_loaded.model_dump(mode="json") == config.model_dump(mode="json")
    assert json_loaded.model_dump(mode="json") == config.model_dump(mode="json")


def test_gui_config_to_report_config_preserves_custom_band() -> None:
    config = GuiReportConfig()
    config.lfp_view.band_preset = "custom"
    config.lfp_view.custom_band_low_hz = 17.0
    config.lfp_view.custom_band_high_hz = 29.0
    config.report.render.include_cover_page = False

    report_config = config.to_report_config()

    assert report_config.bands.default_primary == "custom"
    assert report_config.bands.definitions["custom"].low_hz == 17.0
    assert report_config.bands.definitions["custom"].high_hz == 29.0
    assert report_config.render.include_cover_page is False
