from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from .config import ReportConfig
from .model import DepthSummaryRow, UnitSummary


def apply_matplotlib_defaults() -> None:
    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.titlesize": 12,
            "font.size": 9,
            "grid.alpha": 0.2,
            "legend.fontsize": 8,
            "savefig.bbox": "tight",
        }
    )


def add_placeholder(ax: Axes, title: str, reason: str, config: ReportConfig) -> None:
    ax.set_title(title)
    ax.axis("off")
    ax.text(
        0.5,
        0.58,
        config.render.placeholder_text,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(0.5, 0.42, reason, ha="center", va="center", fontsize=9, transform=ax.transAxes)


def _depth_ticks(depths_mm: Sequence[float]) -> np.ndarray:
    depths = np.asarray(depths_mm, dtype=np.float64)
    if depths.size <= 8:
        return depths
    idx = np.linspace(0, depths.size - 1, min(8, depths.size), dtype=int)
    return depths[idx]


def _frequency_ticks(freq_hz: np.ndarray, count: int = 6) -> tuple[np.ndarray, list[str]]:
    freq = np.asarray(freq_hz, dtype=np.float64)
    if freq.size == 0:
        return np.zeros(0, dtype=np.float64), []
    raw = np.linspace(float(freq[0]), float(freq[-1]), max(count, 2))
    labels = np.ceil(raw).astype(int)
    positions: list[float] = []
    formatted: list[str] = []
    seen: set[int] = set()
    for position, label in zip(raw, labels, strict=False):
        if int(label) in seen:
            continue
        seen.add(int(label))
        positions.append(float(position))
        formatted.append(str(int(label)))
    return np.asarray(positions, dtype=np.float64), formatted


def plot_spike_raster(
    ax: Axes,
    depths_mm: Sequence[float],
    spike_times_by_depth: Sequence[np.ndarray],
    config: ReportConfig,
) -> None:
    valid: list[np.ndarray] = []
    offsets: list[float] = []
    max_time = 0.0
    for depth, times in zip(depths_mm, spike_times_by_depth, strict=False):
        arr = np.asarray(times, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if config.render.raster_window_s > 0:
            arr = arr[(arr >= 0.0) & (arr <= config.render.raster_window_s)]
        if arr.size:
            valid.append(arr)
            offsets.append(float(depth))
            max_time = max(max_time, float(np.max(arr)))
    if not valid:
        add_placeholder(ax, "Spike Raster vs Depth", "No spike times available", config)
        return
    ax.eventplot(
        valid,
        orientation="horizontal",
        lineoffsets=offsets,
        linelengths=0.7,
        linewidths=0.8,
        colors="#1a1a1a",
    )
    ax.set_title("Spike Raster vs Depth")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth (mm)")
    upper = config.render.raster_window_s if config.render.raster_window_s > 0 else max_time
    ax.set_xlim(0.0, max(upper, 0.1))
    ticks = _depth_ticks(depths_mm)
    ax.set_yticks(ticks)
    ax.set_ylim(float(np.min(depths_mm)) - 0.5, float(np.max(depths_mm)) + 0.5)


def plot_lfp_heatmap(
    ax: Axes,
    depths_mm: Sequence[float],
    freq_hz: np.ndarray,
    psd_db_by_depth: np.ndarray,
    band_limits_hz: tuple[float, float],
    band_name: str,
    config: ReportConfig,
) -> None:
    if psd_db_by_depth.size == 0 or freq_hz.size == 0:
        add_placeholder(ax, "LFP Depth x Frequency", "No LFP PSD available", config)
        return
    depths = np.asarray(depths_mm, dtype=np.float64)
    extent = [float(np.min(depths)), float(np.max(depths)), float(freq_hz[0]), float(freq_hz[-1])]
    finite = psd_db_by_depth[np.isfinite(psd_db_by_depth)]
    if finite.size:
        norm = Normalize(
            vmin=float(np.nanpercentile(finite, 5)), vmax=float(np.nanpercentile(finite, 95))
        )
    else:
        norm = None
    image = ax.imshow(
        np.asarray(psd_db_by_depth, dtype=np.float64).T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        norm=norm,
    )
    low_hz, high_hz = band_limits_hz
    ax.axhspan(low_hz, high_hz, color="#f4d35e", alpha=0.18)
    ax.set_title("LFP Depth x Frequency")
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Frequency (Hz)")
    tick_positions, tick_labels = _frequency_ticks(freq_hz)
    ax.set_yticks(tick_positions, labels=tick_labels)
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("PSD (dB)")
    ax.text(
        0.99,
        0.97,
        f"{band_name}: {low_hz:.0f}-{high_hz:.0f} Hz",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
    )


def plot_bandpower_profile(
    ax: Axes,
    depths_mm: Sequence[float],
    values_db: Sequence[float | None],
    band_name: str,
    band_limits_hz: tuple[float, float],
    config: ReportConfig,
) -> None:
    depths = np.asarray(depths_mm, dtype=np.float64)
    values = np.asarray(
        [np.nan if value is None else float(value) for value in values_db], dtype=np.float64
    )
    if not np.any(np.isfinite(values)):
        add_placeholder(ax, f"{band_name} Bandpower vs Depth", "No LFP bandpower available", config)
        return
    low_hz, high_hz = band_limits_hz
    ax.plot(depths, values, marker="o", lw=1.4, color="#0b7189")
    ax.set_title(f"{band_name} Bandpower vs Depth")
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Bandpower (dB)")
    ax.axhline(np.nanmedian(values), color="#999999", linestyle="--", linewidth=0.8)
    ax.text(
        0.98,
        0.96,
        f"{low_hz:.0f}-{high_hz:.0f} Hz",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
    )


def _metric_values(units: Sequence[UnitSummary], metric: str) -> tuple[np.ndarray, np.ndarray]:
    depths = np.asarray([unit.depth_mm for unit in units], dtype=np.float64)
    values = np.asarray([getattr(unit, metric) for unit in units], dtype=np.float64)
    mask = np.isfinite(depths) & np.isfinite(values)
    return depths[mask], values[mask]


def plot_unit_metric_scatter(
    ax: Axes,
    units: Sequence[UnitSummary],
    metric: str,
    title: str,
    ylabel: str,
    color: str,
    config: ReportConfig,
) -> None:
    depths, values = _metric_values(units, metric)
    if values.size == 0:
        add_placeholder(ax, title, "No unit metrics available", config)
        return
    ax.scatter(depths, values, s=20, alpha=0.8, color=color, edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel(ylabel)


def plot_unit_metric_hist2d(
    ax: Axes,
    units: Sequence[UnitSummary],
    metric: str,
    title: str,
    ylabel: str,
    config: ReportConfig,
) -> None:
    depths, values = _metric_values(units, metric)
    if values.size == 0:
        add_placeholder(ax, title, "No unit metrics available", config)
        return
    bins_x = min(12, max(4, len(np.unique(depths))))
    bins_y = min(12, max(4, int(np.sqrt(values.size))))
    hist = ax.hist2d(depths, values, bins=[bins_x, bins_y], cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel(ylabel)
    colorbar = ax.figure.colorbar(hist[3], ax=ax, pad=0.01)
    colorbar.set_label("Count")


def render_summary_table(
    ax: Axes,
    rows: Sequence[DepthSummaryRow],
    primary_band: str,
    amplitude_label: str,
    *,
    font_size: int = 7,
    scale_y: float = 1.2,
) -> None:
    ax.axis("off")
    headers = [
        "Depth",
        f"{primary_band} dB",
        "Spikes",
        "Units",
        "FR Hz",
        f"Amp {amplitude_label}",
        "ISI",
        "Presence",
        "Cutoff",
        "RMS",
    ]
    body: list[list[str]] = []
    for row in rows:
        body.append(
            [
                f"{row.depth_mm:.2f}",
                _fmt(row.primary_band_power_db),
                str(row.spike_count),
                str(row.n_units),
                _fmt(row.median_fr_hz),
                _fmt(row.median_amp_p2p),
                _fmt(row.median_isi_violation_ratio),
                _fmt(row.median_presence_ratio),
                _fmt(row.median_amplitude_cutoff_proxy),
                _fmt(row.mer_rms),
            ]
        )
    table = ax.table(cellText=body, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, scale_y)


def plot_multiband_profiles(
    fig: Figure,
    depths_mm: Sequence[float],
    band_profiles: dict[str, Sequence[float | None]],
    config: ReportConfig,
) -> None:
    bands = list(band_profiles)
    if not bands:
        fig.text(0.5, 0.5, config.render.placeholder_text, ha="center", va="center")
        return
    nrows = len(bands)
    axes = fig.subplots(nrows=nrows, ncols=1, squeeze=False)
    for ax, band_name in zip(axes[:, 0], bands, strict=False):
        values = band_profiles[band_name]
        numeric = np.asarray(
            [np.nan if item is None else float(item) for item in values], dtype=np.float64
        )
        if not np.any(np.isfinite(numeric)):
            add_placeholder(
                ax, f"{band_name} Bandpower vs Depth", "No LFP bandpower available", config
            )
            continue
        ax.plot(depths_mm, numeric, marker="o", lw=1.4, color="#7f5539")
        ax.set_title(f"{band_name} Bandpower vs Depth")
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("Bandpower (dB)")
    fig.tight_layout()


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if not np.isfinite(value):
        return "-"
    return f"{value:.2f}"
