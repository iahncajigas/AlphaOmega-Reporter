from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtWidgets

from alphaomega_reporter import ReportConfig
from alphaomega_reporter.viz import (
    add_placeholder,
    apply_matplotlib_defaults,
    plot_bandpower_profile,
    plot_lfp_heatmap,
    plot_mua_rms_profile,
    plot_spike_raster,
)


class MatplotlibWidget(QtWidgets.QWidget):
    def __init__(
        self, parent: QtWidgets.QWidget | None = None, *, with_toolbar: bool = True
    ) -> None:
        super().__init__(parent)
        apply_matplotlib_defaults()
        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self) if with_toolbar else None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if self.toolbar is not None:
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def _reset(self):
        self.figure.clear()
        return self.figure.add_subplot(111)

    def draw_placeholder(self, title: str, reason: str, config: ReportConfig) -> None:
        ax = self._reset()
        add_placeholder(ax, title, reason, config)
        self.canvas.draw_idle()

    def draw_signal(
        self,
        times_s: np.ndarray,
        values: np.ndarray,
        *,
        title: str,
        ylabel: str,
        config: ReportConfig,
    ) -> None:
        ax = self._reset()
        if values.size == 0:
            add_placeholder(ax, title, "No signal available", config)
        else:
            ax.plot(times_s, values, color="#264653", linewidth=0.8)
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
        self.canvas.draw_idle()

    def draw_raster(
        self,
        depths_mm: Sequence[float],
        spike_times: Sequence[np.ndarray],
        config: ReportConfig,
    ) -> None:
        ax = self._reset()
        plot_spike_raster(ax, depths_mm, spike_times, config)
        self.canvas.draw_idle()

    def draw_waveforms(
        self,
        waveform_time_ms: np.ndarray | None,
        mean_waveforms: dict[int, np.ndarray],
        config: ReportConfig,
    ) -> None:
        ax = self._reset()
        if waveform_time_ms is None or not mean_waveforms:
            add_placeholder(ax, "Waveforms", "No unit waveforms available", config)
        else:
            for unit_id, waveform in sorted(mean_waveforms.items()):
                ax.plot(waveform_time_ms, waveform, linewidth=1.0, label=f"Unit {unit_id}")
            ax.set_title("Mean Waveforms")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="best")
        self.canvas.draw_idle()

    def draw_heatmap(
        self,
        depths_mm: Sequence[float],
        freq_hz: np.ndarray,
        psd_db_by_depth: np.ndarray,
        band_limits_hz: tuple[float, float],
        band_name: str,
        config: ReportConfig,
    ) -> None:
        ax = self._reset()
        plot_lfp_heatmap(ax, depths_mm, freq_hz, psd_db_by_depth, band_limits_hz, band_name, config)
        self.canvas.draw_idle()

    def draw_bandpower(
        self,
        depths_mm: Sequence[float],
        values_db: Sequence[float | None],
        band_name: str,
        band_limits_hz: tuple[float, float],
        config: ReportConfig,
    ) -> None:
        ax = self._reset()
        plot_bandpower_profile(ax, depths_mm, values_db, band_name, band_limits_hz, config)
        self.canvas.draw_idle()

    def draw_mua_rms(
        self,
        depths_mm: Sequence[float],
        rms_values: Sequence[float | None],
        mua_values: Sequence[float | None],
        config: ReportConfig,
    ) -> None:
        ax = self._reset()
        plot_mua_rms_profile(ax, depths_mm, rms_values, mua_values, config)
        self.canvas.draw_idle()
