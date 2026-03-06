from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from alphaomega_reporter import ReportConfig
from alphaomega_reporter.config import BandDefinition


class MerViewConfig(BaseModel):
    highpass_hz: float = 300.0
    lowpass_hz: float | None = None
    notch_hz: float | None = None
    notch_width_hz: float = 4.0
    filter_order: int = 3
    preview_seconds: float = 1.0
    threshold_mad: float = 4.5
    refractory_ms: float = 1.5
    run_fast_sorting: bool = False

    @model_validator(mode="after")
    def _validate(self) -> MerViewConfig:
        if self.lowpass_hz is not None and self.lowpass_hz <= 0:
            raise ValueError("lowpass_hz must be > 0 when set")
        if self.notch_hz is not None and self.notch_hz <= 0:
            raise ValueError("notch_hz must be > 0 when set")
        if self.notch_width_hz <= 0:
            raise ValueError("notch_width_hz must be > 0")
        if self.filter_order <= 0:
            raise ValueError("filter_order must be > 0")
        if self.preview_seconds <= 0:
            raise ValueError("preview_seconds must be > 0")
        if self.threshold_mad <= 0:
            raise ValueError("threshold_mad must be > 0")
        if self.refractory_ms <= 0:
            raise ValueError("refractory_ms must be > 0")
        return self


class LfpViewConfig(BaseModel):
    psd_method: str = "welch"
    band_preset: str = "beta"
    custom_band_low_hz: float = 13.0
    custom_band_high_hz: float = 30.0
    freq_min_hz: float = 0.0
    freq_max_hz: float = 100.0
    notch_hz: float | None = None
    notch_width_hz: float = 4.0

    @model_validator(mode="after")
    def _validate(self) -> LfpViewConfig:
        if self.psd_method not in {"welch", "multitaper"}:
            raise ValueError("psd_method must be 'welch' or 'multitaper'")
        if self.custom_band_low_hz < 0:
            raise ValueError("custom_band_low_hz must be >= 0")
        if self.custom_band_high_hz <= self.custom_band_low_hz:
            raise ValueError("custom_band_high_hz must be > custom_band_low_hz")
        if self.freq_max_hz <= self.freq_min_hz:
            raise ValueError("freq_max_hz must be > freq_min_hz")
        if self.notch_hz is not None and self.notch_hz <= 0:
            raise ValueError("notch_hz must be > 0 when set")
        if self.notch_width_hz <= 0:
            raise ValueError("notch_width_hz must be > 0")
        return self


class GuiReportConfig(BaseModel):
    report: ReportConfig = Field(default_factory=ReportConfig)
    mer: MerViewConfig = Field(default_factory=MerViewConfig)
    lfp_view: LfpViewConfig = Field(default_factory=LfpViewConfig)
    output_pdf: str | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> GuiReportConfig:
        config_path = Path(path)
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text) or {}
        return cls.model_validate(data)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        if target.suffix.lower() == ".json":
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return target

    def to_report_config(self) -> ReportConfig:
        report = ReportConfig.model_validate(self.report.model_dump(mode="json"))
        if self.lfp_view.band_preset == "custom":
            report.bands.definitions["custom"] = BandDefinition(
                low_hz=self.lfp_view.custom_band_low_hz,
                high_hz=self.lfp_view.custom_band_high_hz,
            )
            report.bands.default_primary = "custom"
        else:
            report.bands.default_primary = self.lfp_view.band_preset
        report.lfp.fmin_hz = self.lfp_view.freq_min_hz
        report.lfp.fmax_hz = self.lfp_view.freq_max_hz
        report.lfp.noise_notch_hz = self.lfp_view.notch_hz
        report.lfp.noise_notch_width_hz = self.lfp_view.notch_width_hz
        report.render.raster_window_s = self.mer.preview_seconds
        report.sorting.highpass_hz = self.mer.highpass_hz
        report.sorting.threshold_mad = self.mer.threshold_mad
        report.qc.refractory_ms = self.mer.refractory_ms
        return report
