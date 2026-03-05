from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .config import ReportConfig
from .features import (
    PSDResult,
    band_limits,
    compute_bandpower_db,
    compute_mer_rms,
    compute_welch_psd,
    interpolate_psd_to_grid,
    ordered_bands,
)
from .io import (
    AlphaOmegaReporterError,
    build_trajectory_groups,
    load_case,
    select_lfp_stream,
    select_mer_stream,
)
from .model import DepthSummaryRow, Segment, Session, TrajectoryGroup, UnitSummary
from .qc import amplitude_cutoff_proxy, isi_violation_ratio, presence_ratio
from .sorting_fast import FastSortingResult, sort_single_channel
from .sorting_spikeinterface import sort_with_spikeinterface
from .viz import (
    add_placeholder,
    apply_matplotlib_defaults,
    plot_bandpower_profile,
    plot_lfp_heatmap,
    plot_multiband_profiles,
    plot_spike_raster,
    plot_unit_metric_hist2d,
    plot_unit_metric_scatter,
    render_summary_table,
)

LOGGER = logging.getLogger("alphaomega_reporter")
MAX_ROWS_ON_PLOT_PAGE = 20
MAX_ROWS_PER_TABLE_PAGE = 28


@dataclass
class DepthAnalysis:
    segment: Segment
    depth_mm: float
    raster_times_s: np.ndarray
    unit_summaries: list[UnitSummary]
    summary_row: DepthSummaryRow
    lfp_psd: PSDResult | None
    warnings: list[str]
    amplitude_units: str


@dataclass
class TrajectoryAnalysis:
    group: TrajectoryGroup
    depth_items: list[DepthAnalysis]
    freq_hz: np.ndarray
    psd_db_by_depth: np.ndarray
    band_profiles_db: dict[str, list[float | None]]
    unit_summaries: list[UnitSummary]
    warnings: list[str]
    amplitude_units: str


def build_report(case_dir: Path | str, out: Path | str, config: ReportConfig | None = None) -> Path:
    report_config = config or ReportConfig()
    session = load_case(case_dir, report_config)
    return build_report_from_session(session, out, report_config)


def build_report_from_session(
    session: Session, out: Path | str, config: ReportConfig | None = None
) -> Path:
    report_config = config or ReportConfig()
    apply_matplotlib_defaults()
    groups = build_trajectory_groups(session, report_config)
    if not groups:
        raise AlphaOmegaReporterError(
            "No trajectory groups with side, trajectory, and depth metadata were found"
        )

    primary_band = report_config.bands.default_primary
    plotted_bands = ordered_bands(primary_band, report_config.render.plot_bands, report_config)
    analyses = [_analyze_trajectory(group, plotted_bands, report_config) for group in groups]

    out_path = Path(out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        summary_fig = _case_summary_figure(session, analyses, plotted_bands, report_config)
        pdf.savefig(summary_fig)
        plt.close(summary_fig)
        for key, grouped_analyses in _group_analyses_for_render(analyses, report_config):
            if report_config.render.group_by != "trajectory":
                group_fig = _group_cover_figure(key, grouped_analyses, report_config)
                pdf.savefig(group_fig)
                plt.close(group_fig)
            for analysis in grouped_analyses:
                page1, page2, page3, table_pages = _trajectory_figures(
                    analysis, plotted_bands, report_config
                )
                pdf.savefig(page1)
                pdf.savefig(page2)
                if page3 is not None:
                    pdf.savefig(page3)
                for table_page in table_pages:
                    pdf.savefig(table_page)
                plt.close(page1)
                plt.close(page2)
                if page3 is not None:
                    plt.close(page3)
                for table_page in table_pages:
                    plt.close(table_page)
    LOGGER.info("Wrote report to %s", out_path)
    return out_path


def _group_analyses_for_render(
    analyses: list[TrajectoryAnalysis],
    config: ReportConfig,
) -> list[tuple[str, list[TrajectoryAnalysis]]]:
    mode = config.render.group_by
    if mode == "trajectory":
        return [
            (f"{analysis.group.side}T{analysis.group.trajectory}", [analysis])
            for analysis in analyses
        ]
    buckets: dict[str, list[TrajectoryAnalysis]] = {}
    for analysis in analyses:
        if mode == "side":
            key = analysis.group.side
        else:
            if not analysis.group.target:
                raise AlphaOmegaReporterError(
                    "group-by target requested but no target resolved for "
                    f"{analysis.group.side}T{analysis.group.trajectory}"
                )
            key = analysis.group.target
        buckets.setdefault(key, []).append(analysis)
    return [(key, buckets[key]) for key in sorted(buckets)]


def _trajectory_figures(
    analysis: TrajectoryAnalysis,
    plotted_bands: list[str],
    config: ReportConfig,
) -> tuple[plt.Figure, plt.Figure, plt.Figure | None, list[plt.Figure]]:
    primary_band = config.bands.default_primary
    fig1 = plt.figure(figsize=(11, 8.5), dpi=config.render.dpi, constrained_layout=True)
    outer = fig1.add_gridspec(1, 2, width_ratios=[1.0, 1.2])
    ax_raster = fig1.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.6, 1.0])
    ax_heatmap = fig1.add_subplot(right[0, 0])
    ax_band = fig1.add_subplot(right[1, 0])
    fig1.suptitle(_trajectory_title(analysis, plotted_bands, config))

    plot_spike_raster(
        ax_raster,
        [item.depth_mm for item in analysis.depth_items],
        [item.raster_times_s for item in analysis.depth_items],
        config,
    )
    if analysis.freq_hz.size and analysis.psd_db_by_depth.size:
        plot_lfp_heatmap(
            ax_heatmap,
            [item.depth_mm for item in analysis.depth_items],
            analysis.freq_hz,
            analysis.psd_db_by_depth,
            band_limits(primary_band, config),
            primary_band,
            config,
        )
    else:
        add_placeholder(ax_heatmap, "LFP Depth x Frequency", "No LFP PSD available", config)
    plot_bandpower_profile(
        ax_band,
        [item.depth_mm for item in analysis.depth_items],
        analysis.band_profiles_db.get(primary_band, []),
        primary_band,
        band_limits(primary_band, config),
        config,
    )

    dense_table = len(analysis.depth_items) > MAX_ROWS_ON_PLOT_PAGE
    fig2 = plt.figure(figsize=(11, 8.5), dpi=config.render.dpi)
    if dense_table:
        grid = fig2.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    else:
        grid = fig2.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.3])
    ax_fr_scatter = fig2.add_subplot(grid[0, 0])
    ax_fr_hist = fig2.add_subplot(grid[0, 1])
    ax_amp_scatter = fig2.add_subplot(grid[1, 0])
    ax_amp_hist = fig2.add_subplot(grid[1, 1])
    ax_table = fig2.add_subplot(grid[2, :]) if not dense_table else None
    fig2.suptitle(_trajectory_title(analysis, plotted_bands, config))
    plot_unit_metric_scatter(
        ax_fr_scatter,
        analysis.unit_summaries,
        "firing_rate_hz",
        "Firing Rate vs Depth",
        "Firing Rate (Hz)",
        "#2b9348",
        config,
    )
    plot_unit_metric_hist2d(
        ax_fr_hist,
        analysis.unit_summaries,
        "firing_rate_hz",
        "Firing Rate 2D Histogram",
        "Firing Rate (Hz)",
        config,
    )
    plot_unit_metric_scatter(
        ax_amp_scatter,
        analysis.unit_summaries,
        "median_p2p",
        "Waveform Amplitude vs Depth",
        f"Waveform p2p ({analysis.amplitude_units})",
        "#9d4edd",
        config,
    )
    plot_unit_metric_hist2d(
        ax_amp_hist,
        analysis.unit_summaries,
        "median_p2p",
        "Waveform Amplitude 2D Histogram",
        f"Waveform p2p ({analysis.amplitude_units})",
        config,
    )
    if ax_table is not None:
        render_summary_table(
            ax_table,
            [item.summary_row for item in analysis.depth_items],
            primary_band,
            analysis.amplitude_units,
        )
    else:
        fig2.text(
            0.01,
            0.06,
            (
                "Per-depth summary table moved to the following page(s) "
                f"because this trajectory has {len(analysis.depth_items)} depths."
            ),
            ha="left",
            va="bottom",
            fontsize=8,
        )
    warning_text = _format_warnings(analysis)
    fig2.text(0.01, 0.02, warning_text, ha="left", va="bottom", fontsize=8)
    fig2.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.92,
        bottom=0.08,
        hspace=0.55,
        wspace=0.25,
    )

    fig3: plt.Figure | None = None
    if len(plotted_bands) > 1:
        fig3 = plt.figure(figsize=(11, 8.5), dpi=config.render.dpi)
        fig3.suptitle(_trajectory_title(analysis, plotted_bands, config))
        plot_multiband_profiles(
            fig3,
            [item.depth_mm for item in analysis.depth_items],
            {band: analysis.band_profiles_db.get(band, []) for band in plotted_bands},
            config,
        )
    table_pages = _summary_table_pages(analysis, config) if dense_table else []
    return fig1, fig2, fig3, table_pages


def _summary_table_pages(
    analysis: TrajectoryAnalysis,
    config: ReportConfig,
) -> list[plt.Figure]:
    rows = [item.summary_row for item in analysis.depth_items]
    total_pages = int(np.ceil(len(rows) / MAX_ROWS_PER_TABLE_PAGE))
    figures: list[plt.Figure] = []
    for page_index, chunk in enumerate(_chunked(rows, MAX_ROWS_PER_TABLE_PAGE), start=1):
        fig = plt.figure(figsize=(11, 8.5), dpi=config.render.dpi)
        ax = fig.add_subplot(111)
        fig.suptitle(
            f"{_trajectory_title(analysis, [], config)} | table {page_index}/{total_pages}",
            fontsize=11,
        )
        render_summary_table(
            ax,
            chunk,
            config.bands.default_primary,
            analysis.amplitude_units,
            font_size=8,
            scale_y=1.4,
        )
        fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
        figures.append(fig)
    return figures


def _trajectory_title(
    analysis: TrajectoryAnalysis, plotted_bands: list[str], config: ReportConfig
) -> str:
    group = analysis.group
    target = group.target or "Unknown target"
    return (
        f"{group.side}T{group.trajectory} | {target} | "
        f"primary={config.bands.default_primary} | plotted={', '.join(plotted_bands)} | "
        f"sort={config.sorting.mode}"
    )


def _case_summary_figure(
    session: Session,
    analyses: list[TrajectoryAnalysis],
    plotted_bands: list[str],
    config: ReportConfig,
) -> plt.Figure:
    fig = plt.figure(figsize=(11, 8.5), dpi=config.render.dpi)
    ax = fig.add_subplot(111)
    ax.axis("off")
    case_meta = session.meta.get("case_metadata", {})
    lines = [
        f"Case: {session.meta.get('case_name', Path(session.session_path).name)}",
        f"Source: {session.session_path}",
        f"Segments: {len(session.segments)}",
        f"Trajectories: {len(analyses)}",
        f"Grouping: {config.render.group_by}",
        f"Primary band: {config.bands.default_primary}",
        f"Plotted bands: {', '.join(plotted_bands)}",
        f"Sort mode: {config.sorting.mode}",
        f"XML sidecar: {case_meta.get('xml_path') or 'not found'}",
        f"Case target: {case_meta.get('target_case_default') or 'unresolved'}",
    ]
    ax.text(0.05, 0.95, "AlphaOmega Reporter Summary", fontsize=16, fontweight="bold", va="top")
    ax.text(0.05, 0.86, "\n".join(lines), va="top", family="monospace")
    listing = []
    for analysis in analyses:
        listing.append(
            f"{analysis.group.side}T{analysis.group.trajectory}"
            f" target={analysis.group.target or '?'}"
            f" depths={len(analysis.depth_items)}"
            f" warnings={len(analysis.warnings)}"
        )
    ax.text(0.05, 0.46, "Trajectories", fontsize=12, fontweight="bold")
    ax.text(0.05, 0.42, "\n".join(listing), va="top", family="monospace")
    return fig


def _group_cover_figure(
    key: str,
    analyses: list[TrajectoryAnalysis],
    config: ReportConfig,
) -> plt.Figure:
    fig = plt.figure(figsize=(11, 8.5), dpi=config.render.dpi)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.82, config.render.group_by.title(), ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.68, key, ha="center", va="center", fontsize=24, fontweight="bold")
    lines = [
        f"{analysis.group.side}T{analysis.group.trajectory} "
        f"target={analysis.group.target or '?'} "
        f"depths={len(analysis.depth_items)}"
        for analysis in analyses
    ]
    ax.text(0.5, 0.48, "\n".join(lines), ha="center", va="center", family="monospace")
    return fig


def _analyze_trajectory(
    group: TrajectoryGroup,
    plotted_bands: list[str],
    config: ReportConfig,
) -> TrajectoryAnalysis:
    depth_items: list[DepthAnalysis] = []
    all_units: list[UnitSummary] = []
    all_warnings: list[str] = []
    primary_band = config.bands.default_primary
    amplitude_units = "a.u."
    for segment in group.segments:
        analysis = _analyze_segment(segment, plotted_bands, config)
        depth_items.append(analysis)
        all_units.extend(analysis.unit_summaries)
        all_warnings.extend(analysis.warnings)
        if analysis.amplitude_units:
            amplitude_units = analysis.amplitude_units

    ref_freq = np.zeros(0, dtype=np.float64)
    for item in depth_items:
        if item.lfp_psd is not None and item.lfp_psd.freq_hz.size:
            ref_freq = item.lfp_psd.freq_hz
            break
    if ref_freq.size:
        rows = []
        for item in depth_items:
            if item.lfp_psd is None:
                rows.append(np.full(ref_freq.shape, np.nan, dtype=np.float64))
                continue
            rows.append(
                10.0
                * np.log10(
                    interpolate_psd_to_grid(item.lfp_psd.freq_hz, item.lfp_psd.psd, ref_freq)
                    + np.finfo(float).eps
                )
            )
        psd_db_by_depth = np.vstack(rows)
    else:
        psd_db_by_depth = np.zeros((0, 0), dtype=np.float64)

    band_profiles: dict[str, list[float | None]] = {}
    for band_name in plotted_bands:
        values: list[float | None] = []
        for item in depth_items:
            value = (
                item.summary_row.primary_band_power_db
                if band_name == primary_band
                else item.summary_row.extra_band_powers_db.get(band_name)
            )
            values.append(value if value is not None and np.isfinite(value) else None)
        band_profiles[band_name] = values

    return TrajectoryAnalysis(
        group=group,
        depth_items=depth_items,
        freq_hz=ref_freq,
        psd_db_by_depth=psd_db_by_depth,
        band_profiles_db=band_profiles,
        unit_summaries=all_units,
        warnings=sorted(set(group.warnings + all_warnings)),
        amplitude_units=amplitude_units,
    )


def _analyze_segment(
    segment: Segment, plotted_bands: list[str], config: ReportConfig
) -> DepthAnalysis:
    depth_mm = float(segment.meta["depth_mm"])
    warnings: list[str] = []

    mer_selection = select_mer_stream(segment, config)
    mer_values = _extract_stream_values(segment, mer_selection)
    mer_units = mer_selection.units if mer_selection is not None and mer_selection.units else "a.u."
    if mer_values is None:
        warnings.append("MER stream unavailable")

    lfp_selection = select_lfp_stream(segment, config)
    lfp_values = _extract_stream_values(segment, lfp_selection)
    if lfp_values is None:
        warnings.append("LFP stream unavailable")

    lfp_psd: PSDResult | None = None
    band_powers_db: dict[str, float] = {}
    if lfp_values is not None and lfp_selection is not None:
        try:
            lfp_psd = compute_welch_psd(
                lfp_values,
                lfp_selection.fs_hz,
                fmin_hz=config.lfp.fmin_hz,
                fmax_hz=config.lfp.fmax_hz,
                nperseg=config.lfp.welch_nperseg,
                noverlap=config.lfp.welch_noverlap,
            )
            for band_name in plotted_bands:
                low_hz, high_hz = band_limits(band_name, config)
                band_powers_db[band_name] = compute_bandpower_db(
                    lfp_psd.freq_hz, lfp_psd.psd, low_hz, high_hz
                )
        except Exception as exc:
            warnings.append(f"LFP PSD failed: {exc}")

    duration_s = _segment_duration_s(segment)
    sorting = _summarize_spikes(
        segment,
        mer_values,
        mer_selection.fs_hz if mer_selection else None,
        depth_mm,
        duration_s,
        config,
    )
    warnings.extend(sorting.warnings)
    row = _build_summary_row(
        depth_mm=depth_mm,
        mer_values=mer_values,
        lfp_available=lfp_psd is not None,
        mer_available=mer_values is not None,
        raster_times_s=sorting.spike_times_s,
        unit_summaries=sorting.unit_summaries,
        warnings=warnings,
        primary_band=config.bands.default_primary,
        band_powers_db=band_powers_db,
    )
    return DepthAnalysis(
        segment=segment,
        depth_mm=depth_mm,
        raster_times_s=sorting.spike_times_s,
        unit_summaries=sorting.unit_summaries,
        summary_row=row,
        lfp_psd=lfp_psd,
        warnings=warnings,
        amplitude_units=mer_units,
    )


def _extract_stream_values(segment: Segment, selection) -> np.ndarray | None:
    if selection is None:
        return None
    stream = segment.streams.get(selection.stream_key)
    if stream is None:
        return None
    data = np.asarray(stream.data)
    if data.ndim == 1:
        return data.astype(np.float64, copy=False)
    if data.shape[1] <= selection.channel_index:
        return None
    return np.asarray(data[:, selection.channel_index], dtype=np.float64)


def _segment_duration_s(segment: Segment) -> float:
    duration = 0.0
    for stream in segment.streams.values():
        if stream.fs_hz > 0:
            duration = max(
                duration, float(stream.t0_s) + (float(stream.data.shape[0]) / float(stream.fs_hz))
            )
    for spike in segment.spikes.values():
        if spike.times_s.size:
            duration = max(duration, float(np.max(spike.times_s)))
    return max(duration, 1.0)


def _summarize_spikes(
    segment: Segment,
    mer_values: np.ndarray | None,
    fs_hz: float | None,
    depth_mm: float,
    duration_s: float,
    config: ReportConfig,
) -> FastSortingResult:
    if config.sorting.mode == "none":
        return _summarize_native_spikes(segment, depth_mm, duration_s, config)
    if mer_values is None or fs_hz is None:
        return FastSortingResult(
            spike_times_s=np.zeros(0, dtype=np.float64),
            labels=np.zeros(0, dtype=np.int32),
            waveforms=np.zeros((0, 0), dtype=np.float32),
            unit_summaries=[],
            warnings=["No MER stream available for spike sorting"],
        )
    if config.sorting.mode == "spikeinterface":
        return sort_with_spikeinterface(mer_values, fs_hz, depth_mm, config)
    return sort_single_channel(mer_values, fs_hz, depth_mm, config)


def _summarize_native_spikes(
    segment: Segment,
    depth_mm: float,
    duration_s: float,
    config: ReportConfig,
) -> FastSortingResult:
    spike = _select_native_spike_stream(segment)
    if spike is None:
        return FastSortingResult(
            spike_times_s=np.zeros(0, dtype=np.float64),
            labels=np.zeros(0, dtype=np.int32),
            waveforms=np.zeros((0, 0), dtype=np.float32),
            unit_summaries=[],
            warnings=["No native spike stream available"],
        )
    times = np.asarray(spike.times_s, dtype=np.float64)
    times = times[np.isfinite(times)]
    labels = np.zeros(times.size, dtype=np.int32)
    unit_summaries: list[UnitSummary] = []
    warnings: list[str] = []
    waveforms = (
        np.asarray(spike.waveforms, dtype=np.float32)
        if spike.waveforms is not None
        else np.zeros((0, 0), dtype=np.float32)
    )
    if spike.unit_ids is not None and len(spike.unit_ids) == len(times):
        labels = np.asarray(spike.unit_ids, dtype=np.int32)
        for unit_id in sorted(np.unique(labels)):
            mask = labels == unit_id
            unit_times = times[mask]
            unit_waveforms = (
                waveforms[mask]
                if waveforms.shape[0] == times.size
                else np.zeros((0, 0), dtype=np.float32)
            )
            p2p = (
                np.ptp(unit_waveforms, axis=1)
                if unit_waveforms.size
                else np.zeros(0, dtype=np.float32)
            )
            unit_summaries.append(
                UnitSummary(
                    depth_mm=depth_mm,
                    unit_id=int(unit_id),
                    spike_times_s=unit_times,
                    firing_rate_hz=float(unit_times.size / max(duration_s, np.finfo(float).eps)),
                    median_p2p=float(np.nanmedian(p2p)) if p2p.size else float("nan"),
                    isi_violation_ratio=isi_violation_ratio(unit_times, config.qc.refractory_ms),
                    presence_ratio=presence_ratio(
                        unit_times, 0.0, duration_s, config.qc.presence_ratio_bins
                    ),
                    amplitude_cutoff_proxy=amplitude_cutoff_proxy(p2p),
                )
            )
    elif times.size:
        warnings.append("Native spike stream lacks unit labels; unit-level metrics unavailable")
    return FastSortingResult(
        spike_times_s=times,
        labels=labels,
        waveforms=waveforms,
        unit_summaries=unit_summaries,
        warnings=warnings,
    )


def _select_native_spike_stream(segment: Segment):
    if not segment.spikes:
        return None
    ranked = []
    for key, spike in segment.spikes.items():
        source = spike.source or {}
        name = str(source.get("channel_name", key)).upper()
        rank = 0 if any(token in name for token in ("SEG", "SPK")) else 1
        ranked.append((rank, -len(spike.times_s), key, spike))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]


def _build_summary_row(
    *,
    depth_mm: float,
    mer_values: np.ndarray | None,
    lfp_available: bool,
    mer_available: bool,
    raster_times_s: np.ndarray,
    unit_summaries: list[UnitSummary],
    warnings: list[str],
    primary_band: str,
    band_powers_db: dict[str, float],
) -> DepthSummaryRow:
    median_fr = _median_or_none([unit.firing_rate_hz for unit in unit_summaries])
    median_amp = _median_or_none([unit.median_p2p for unit in unit_summaries])
    median_isi = _median_or_none([unit.isi_violation_ratio for unit in unit_summaries])
    median_presence = _median_or_none([unit.presence_ratio for unit in unit_summaries])
    median_cutoff = _median_or_none([unit.amplitude_cutoff_proxy for unit in unit_summaries])
    primary_power = band_powers_db.get(primary_band)
    extra = {
        key: value
        for key, value in band_powers_db.items()
        if key != primary_band and np.isfinite(value)
    }
    mer_rms = compute_mer_rms(mer_values) if mer_values is not None else float("nan")
    return DepthSummaryRow(
        depth_mm=depth_mm,
        primary_band_power_db=primary_power
        if primary_power is not None and np.isfinite(primary_power)
        else None,
        spike_count=int(np.asarray(raster_times_s).size),
        n_units=len(unit_summaries),
        median_fr_hz=median_fr,
        median_amp_p2p=median_amp,
        median_isi_violation_ratio=median_isi,
        median_presence_ratio=median_presence,
        median_amplitude_cutoff_proxy=median_cutoff,
        mer_rms=mer_rms if np.isfinite(mer_rms) else None,
        lfp_available=lfp_available,
        mer_available=mer_available,
        warning_flags=sorted(set(warnings)),
        extra_band_powers_db=extra,
    )


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.median(array))


def _format_warnings(analysis: TrajectoryAnalysis) -> str:
    if not analysis.warnings:
        return "Warnings: none"
    return "Warnings: " + " | ".join(sorted(set(analysis.warnings)))


def _chunked(items: list[DepthSummaryRow], size: int) -> Iterable[list[DepthSummaryRow]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
