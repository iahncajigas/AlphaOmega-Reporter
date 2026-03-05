from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from alphaomega_reporter import (
    ReportConfig,
    apply_notch_filter,
    band_limits,
    compute_bandpower_db,
    compute_multitaper_psd,
    compute_welch_psd,
)
from alphaomega_reporter.features import compute_mua_envelope, interpolate_psd_to_grid
from alphaomega_reporter.sorting_fast import detect_spikes_threshold, sort_single_channel

from .models import DepthModel, SegmentModel
from .report_config import GuiReportConfig


@dataclass
class MerPreviewResult:
    times_s: np.ndarray
    values: np.ndarray
    fs_hz: float
    units: str
    raster_depths_mm: list[float]
    raster_spike_times_s: list[np.ndarray]
    waveform_time_ms: np.ndarray | None
    mean_waveforms: dict[int, np.ndarray]
    warnings: list[str] = field(default_factory=list)


@dataclass
class LfpPreviewResult:
    depths_mm: list[float]
    freq_hz: np.ndarray
    psd_db_by_depth: np.ndarray
    band_name: str
    band_limits_hz: tuple[float, float]
    bandpower_db: list[float | None]
    warnings: list[str] = field(default_factory=list)


class ProcessingCache:
    def __init__(self) -> None:
        self.filtered: dict[tuple, np.ndarray] = {}
        self.psd: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def clear(self) -> None:
        self.filtered.clear()
        self.psd.clear()


def _butter_filter(
    values: np.ndarray,
    fs_hz: float,
    *,
    highpass_hz: float | None,
    lowpass_hz: float | None,
    order: int,
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size < 8:
        return data
    nyquist = float(fs_hz) / 2.0
    if highpass_hz and lowpass_hz and 0 < highpass_hz < lowpass_hz < nyquist:
        cutoff = [float(highpass_hz), float(lowpass_hz)]
        btype = "bandpass"
    elif highpass_hz and 0 < highpass_hz < nyquist:
        cutoff = float(highpass_hz)
        btype = "highpass"
    elif lowpass_hz and 0 < lowpass_hz < nyquist:
        cutoff = float(lowpass_hz)
        btype = "lowpass"
    else:
        return data.astype(np.float64, copy=True)
    b, a = signal.butter(max(int(order), 1), cutoff, btype=btype, fs=float(fs_hz))
    padlen = 3 * (max(len(a), len(b)) - 1)
    if data.size <= padlen:
        return data.astype(np.float64, copy=True)
    return signal.filtfilt(b, a, data).astype(np.float64, copy=False)


def _downsample_for_display(values: np.ndarray, max_points: int = 5000) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size <= max_points:
        return data
    step = int(np.ceil(data.size / max_points))
    return data[::step]


def _time_axis(n_samples: int, duration_s: float) -> np.ndarray:
    if n_samples <= 1:
        return np.zeros(max(n_samples, 0), dtype=np.float64)
    return np.linspace(0.0, float(duration_s), n_samples, endpoint=False, dtype=np.float64)


def _filter_mer_values(
    segment_model: SegmentModel,
    channel_index: int,
    gui_config: GuiReportConfig,
    cache: ProcessingCache,
) -> tuple[np.ndarray | None, float | None, str]:
    values, fs_hz, units = segment_model.extract_mer(channel_index)
    if values is None or fs_hz is None:
        return None, None, units
    key = (
        "mer",
        segment_model.segment_id,
        int(channel_index),
        float(fs_hz),
        gui_config.mer.highpass_hz,
        gui_config.mer.lowpass_hz,
        gui_config.mer.notch_hz,
        gui_config.mer.notch_width_hz,
        gui_config.mer.filter_order,
    )
    if key not in cache.filtered:
        filtered = _butter_filter(
            values,
            fs_hz,
            highpass_hz=gui_config.mer.highpass_hz,
            lowpass_hz=gui_config.mer.lowpass_hz,
            order=gui_config.mer.filter_order,
        )
        filtered = apply_notch_filter(
            filtered,
            fs_hz,
            gui_config.mer.notch_hz,
            gui_config.mer.notch_width_hz,
        )
        cache.filtered[key] = filtered
    return cache.filtered[key], fs_hz, units


def _select_native_spikes(
    segment_model: SegmentModel,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    best = None
    for spike in segment_model.segment.spikes.values():
        times = np.asarray(spike.times_s, dtype=np.float64)
        if best is None or times.size > best[0].size:
            best = (times, spike.unit_ids, spike.waveforms)
    if best is None:
        return np.zeros(0, dtype=np.float64), None, None
    return best


def _waveform_means_from_native(
    unit_ids: np.ndarray | None, waveforms: np.ndarray | None
) -> dict[int, np.ndarray]:
    if unit_ids is None or waveforms is None or len(unit_ids) != len(waveforms):
        return {}
    means: dict[int, np.ndarray] = {}
    labels = np.asarray(unit_ids, dtype=np.int32)
    data = np.asarray(waveforms, dtype=np.float64)
    for unit_id in np.unique(labels):
        means[int(unit_id)] = np.nanmean(data[labels == unit_id], axis=0)
    return means


def compute_mer_preview(
    current_depth: DepthModel,
    selected_depths: list[DepthModel],
    channel_index: int,
    gui_config: GuiReportConfig,
    cache: ProcessingCache,
) -> MerPreviewResult:
    values, fs_hz, units = _filter_mer_values(
        current_depth.segment_model, channel_index, gui_config, cache
    )
    if values is None or fs_hz is None:
        return MerPreviewResult(
            times_s=np.zeros(0, dtype=np.float64),
            values=np.zeros(0, dtype=np.float64),
            fs_hz=1.0,
            units=units,
            raster_depths_mm=[],
            raster_spike_times_s=[],
            waveform_time_ms=None,
            mean_waveforms={},
            warnings=["No MER stream found"],
        )
    n_samples = int(round(gui_config.mer.preview_seconds * fs_hz))
    snippet_source = values[: max(n_samples, 1)]
    snippet = _downsample_for_display(snippet_source)
    snippet_times = _time_axis(snippet.size, snippet_source.size / float(fs_hz))

    native_times, native_unit_ids, native_waveforms = _select_native_spikes(
        current_depth.segment_model
    )
    if native_times.size:
        mean_waveforms = _waveform_means_from_native(native_unit_ids, native_waveforms)
        waveform_time_ms = None
        if mean_waveforms:
            first = next(iter(mean_waveforms.values()))
            waveform_time_ms = np.arange(first.size, dtype=np.float64) / fs_hz * 1000.0
    else:
        spike_idx = detect_spikes_threshold(
            values,
            fs_hz,
            gui_config.mer.threshold_mad,
            gui_config.mer.refractory_ms,
        )
        waveform_time_ms = None
        mean_waveforms = {}

    if gui_config.mer.run_fast_sorting:
        report_config = gui_config.to_report_config()
        sorting = sort_single_channel(
            values,
            fs_hz,
            current_depth.depth_mm or 0.0,
            report_config,
        )
        if sorting.waveforms.size:
            waveform_time_ms = (
                np.arange(sorting.waveforms.shape[1], dtype=np.float64) / fs_hz * 1000.0
            )
            mean_waveforms = {}
            for label in np.unique(sorting.labels):
                mean_waveforms[int(label)] = np.nanmean(
                    sorting.waveforms[sorting.labels == label], axis=0
                )

    raster_depths: list[float] = []
    raster_spikes: list[np.ndarray] = []
    warnings: list[str] = []
    for depth_model in selected_depths:
        depth = depth_model.depth_mm
        if depth is None:
            continue
        series, series_fs, _units = _filter_mer_values(
            depth_model.segment_model,
            channel_index,
            gui_config,
            cache,
        )
        if series is None or series_fs is None:
            continue
        native_times, _unit_ids, _waveforms = _select_native_spikes(depth_model.segment_model)
        if native_times.size:
            spike_times = native_times
        else:
            spike_idx = detect_spikes_threshold(
                series,
                series_fs,
                gui_config.mer.threshold_mad,
                gui_config.mer.refractory_ms,
            )
            spike_times = spike_idx.astype(np.float64) / float(series_fs)
        raster_depths.append(float(depth))
        raster_spikes.append(spike_times[spike_times <= gui_config.mer.preview_seconds])
    if not raster_spikes:
        warnings.append("No spike trains available for selected depths")
    return MerPreviewResult(
        times_s=snippet_times,
        values=snippet,
        fs_hz=float(fs_hz),
        units=units,
        raster_depths_mm=raster_depths,
        raster_spike_times_s=raster_spikes,
        waveform_time_ms=waveform_time_ms,
        mean_waveforms=mean_waveforms,
        warnings=warnings,
    )


def _resolve_band(
    gui_config: GuiReportConfig, report_config: ReportConfig
) -> tuple[str, tuple[float, float]]:
    if gui_config.lfp_view.band_preset == "custom":
        return (
            "custom",
            (
                gui_config.lfp_view.custom_band_low_hz,
                gui_config.lfp_view.custom_band_high_hz,
            ),
        )
    return gui_config.lfp_view.band_preset, band_limits(
        gui_config.lfp_view.band_preset, report_config
    )


def compute_lfp_preview(
    selected_depths: list[DepthModel],
    channel_index: int,
    gui_config: GuiReportConfig,
    cache: ProcessingCache,
) -> LfpPreviewResult:
    report_config = gui_config.to_report_config()
    band_name, limits_hz = _resolve_band(gui_config, report_config)
    warnings: list[str] = []
    psd_rows: list[np.ndarray] = []
    depths_mm: list[float] = []
    power_profile: list[float | None] = []
    reference_freq = np.zeros(0, dtype=np.float64)
    raw_psd: list[tuple[np.ndarray, np.ndarray] | None] = []

    for depth_model in selected_depths:
        depth = depth_model.depth_mm
        if depth is None:
            continue
        values, fs_hz, _units = depth_model.segment_model.extract_lfp(channel_index)
        if values is None or fs_hz is None:
            warnings.append(f"No LFP stream for depth {depth:.3f}")
            continue
        filtered = apply_notch_filter(
            values,
            fs_hz,
            gui_config.lfp_view.notch_hz,
            gui_config.lfp_view.notch_width_hz,
        )
        key = (
            "lfp-psd",
            depth_model.segment_model.segment_id,
            int(channel_index),
            float(fs_hz),
            gui_config.lfp_view.psd_method,
            gui_config.lfp_view.freq_min_hz,
            gui_config.lfp_view.freq_max_hz,
            gui_config.lfp_view.notch_hz,
            gui_config.lfp_view.notch_width_hz,
            report_config.lfp.welch_nperseg,
            report_config.lfp.welch_noverlap,
        )
        if key not in cache.psd:
            try:
                if gui_config.lfp_view.psd_method == "multitaper":
                    psd = compute_multitaper_psd(
                        filtered,
                        fs_hz,
                        fmin_hz=gui_config.lfp_view.freq_min_hz,
                        fmax_hz=gui_config.lfp_view.freq_max_hz,
                    )
                else:
                    psd = compute_welch_psd(
                        filtered,
                        fs_hz,
                        fmin_hz=gui_config.lfp_view.freq_min_hz,
                        fmax_hz=gui_config.lfp_view.freq_max_hz,
                        nperseg=report_config.lfp.welch_nperseg,
                        noverlap=report_config.lfp.welch_noverlap,
                    )
                cache.psd[key] = (psd.freq_hz, psd.psd)
            except Exception as exc:
                warnings.append(f"LFP PSD failed at depth {depth:.3f}: {exc}")
                raw_psd.append(None)
                continue
        freq_hz, psd_values = cache.psd[key]
        if reference_freq.size == 0:
            reference_freq = freq_hz
        raw_psd.append((freq_hz, psd_values))
        depths_mm.append(float(depth))
        power = compute_bandpower_db(freq_hz, psd_values, *limits_hz)
        power_profile.append(power if np.isfinite(power) else None)

    if not depths_mm or reference_freq.size == 0:
        return LfpPreviewResult(
            depths_mm=[],
            freq_hz=np.zeros(0, dtype=np.float64),
            psd_db_by_depth=np.zeros((0, 0), dtype=np.float64),
            band_name=band_name,
            band_limits_hz=limits_hz,
            bandpower_db=[],
            warnings=warnings or ["No LFP stream found"],
        )

    for item in raw_psd:
        if item is None:
            continue
        freq_hz, psd_values = item
        interp = interpolate_psd_to_grid(freq_hz, psd_values, reference_freq)
        psd_rows.append(10.0 * np.log10(interp + np.finfo(float).eps))
    if not psd_rows:
        psd_matrix = np.zeros((0, 0), dtype=np.float64)
    else:
        psd_matrix = np.vstack(psd_rows)
    return LfpPreviewResult(
        depths_mm=depths_mm,
        freq_hz=reference_freq,
        psd_db_by_depth=psd_matrix,
        band_name=band_name,
        band_limits_hz=limits_hz,
        bandpower_db=power_profile,
        warnings=warnings,
    )


def compute_mua_profile(
    selected_depths: list[DepthModel],
    channel_index: int,
    gui_config: GuiReportConfig,
    cache: ProcessingCache,
) -> tuple[list[float], list[float | None], list[float | None]]:
    depths: list[float] = []
    rms_values: list[float | None] = []
    mua_values: list[float | None] = []
    for depth_model in selected_depths:
        if depth_model.depth_mm is None:
            continue
        values, fs_hz, _units = _filter_mer_values(
            depth_model.segment_model,
            channel_index,
            gui_config,
            cache,
        )
        if values is None or fs_hz is None:
            continue
        depths.append(float(depth_model.depth_mm))
        rms_values.append(float(np.sqrt(np.mean(np.square(values)))) if values.size else None)
        mua = compute_mua_envelope(values, fs_hz, max(gui_config.mer.highpass_hz, 1.0))
        mua_mean = float(np.nanmean(mua)) if mua.size else float("nan")
        mua_values.append(mua_mean if np.isfinite(mua_mean) else None)
    return depths, rms_values, mua_values
