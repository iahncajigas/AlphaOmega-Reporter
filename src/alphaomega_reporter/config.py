from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class CaseMetadataConfig(BaseModel):
    xml_enabled: bool = True
    xml_glob: str = "*.xml"
    xml_fallback_glob: str = "*.xml.bak"
    ignore_subdirs: list[str] = Field(default_factory=lambda: ["conversion"])
    ignore_suffixes: list[str] = Field(
        default_factory=lambda: [".mat", ".bak", ".bin", ".DS_Store"]
    )


class TargetBinding(BaseModel):
    side: str
    trajectory: int
    target: str

    @field_validator("side")
    @classmethod
    def _normalize_side(cls, value: str) -> str:
        text = str(value).strip().upper()
        if text not in {"L", "R"}:
            raise ValueError("side must be 'L' or 'R'")
        return text


class TargetsConfig(BaseModel):
    case_default_target: str | None = None
    bindings: list[TargetBinding] = Field(default_factory=list)


class DepthRegexRule(BaseModel):
    source: str
    key: str | None = None
    pattern: str | None = None

    @model_validator(mode="after")
    def _validate_rule(self) -> DepthRegexRule:
        if self.source == "meta" and not self.key:
            raise ValueError("meta depth rule requires 'key'")
        if self.source != "meta" and not self.pattern:
            raise ValueError("non-meta depth rule requires 'pattern'")
        return self


class DepthParsingConfig(BaseModel):
    regexes: list[DepthRegexRule] = Field(
        default_factory=lambda: [
            DepthRegexRule(source="meta", key="depth_mm"),
            DepthRegexRule(
                source="segment_id",
                pattern=r"(?i)(?:i)?(?:(?P<hemi>[RL]))?T(?P<traj>\d+)D(?P<depth>[+-]?\d+(?:\.\d+)?)",
            ),
            DepthRegexRule(
                source="source_file",
                pattern=r"(?i)(?:i)?(?:(?P<hemi>[RL]))?T(?P<traj>\d+)D(?P<depth>[+-]?\d+(?:\.\d+)?)",
            ),
        ]
    )


class StreamSelectionConfig(BaseModel):
    mer_name_hints: list[str] = Field(default_factory=lambda: ["SPK", "RAW", "MER", "MICRO"])
    lfp_name_hints: list[str] = Field(default_factory=lambda: ["LFP", "MACRO"])
    prefer_channel_with_max_rms: bool = True


class BandDefinition(BaseModel):
    low_hz: float
    high_hz: float

    @model_validator(mode="after")
    def _validate(self) -> BandDefinition:
        if self.low_hz < 0:
            raise ValueError("band low_hz must be >= 0")
        if self.high_hz <= self.low_hz:
            raise ValueError("band high_hz must be > low_hz")
        return self


class BandsConfig(BaseModel):
    default_primary: str = "beta"
    definitions: dict[str, BandDefinition] = Field(
        default_factory=lambda: {
            "delta": BandDefinition(low_hz=1.0, high_hz=4.0),
            "theta": BandDefinition(low_hz=4.0, high_hz=8.0),
            "alpha": BandDefinition(low_hz=8.0, high_hz=13.0),
            "beta": BandDefinition(low_hz=13.0, high_hz=30.0),
            "highbeta": BandDefinition(low_hz=20.0, high_hz=30.0),
            "gamma": BandDefinition(low_hz=30.0, high_hz=100.0),
        }
    )

    @model_validator(mode="after")
    def _validate_primary(self) -> BandsConfig:
        if self.default_primary not in self.definitions:
            raise ValueError("default_primary must be present in definitions")
        return self


class LfpConfig(BaseModel):
    fmin_hz: float = 0.0
    fmax_hz: float = 100.0
    welch_nperseg: int = 512
    welch_noverlap: int = 256


class SortingConfig(BaseModel):
    mode: str = "fast"
    highpass_hz: float = 300.0
    threshold_mad: float = 4.5
    pre_ms: float = 0.6
    post_ms: float = 1.0
    min_spikes_per_cluster: int = 25
    max_kmeans_clusters: int = 4
    spikeinterface_sorter: str = "tridesclous2"
    common_reference: bool = False


class QcConfig(BaseModel):
    refractory_ms: float = 1.5
    presence_ratio_bins: int = 10


class RenderConfig(BaseModel):
    dpi: int = 150
    raster_window_s: float = 1.0
    placeholder_text: str = "Not available"
    group_by: str = "trajectory"
    plot_bands: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_group_by(self) -> RenderConfig:
        if self.group_by not in {"trajectory", "target", "side"}:
            raise ValueError("group_by must be one of trajectory|target|side")
        return self


class ReportConfig(BaseModel):
    case_metadata: CaseMetadataConfig = Field(default_factory=CaseMetadataConfig)
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    bands: BandsConfig = Field(default_factory=BandsConfig)
    depth_parsing: DepthParsingConfig = Field(default_factory=DepthParsingConfig)
    stream_selection: StreamSelectionConfig = Field(default_factory=StreamSelectionConfig)
    lfp: LfpConfig = Field(default_factory=LfpConfig)
    sorting: SortingConfig = Field(default_factory=SortingConfig)
    qc: QcConfig = Field(default_factory=QcConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> ReportConfig:
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)
