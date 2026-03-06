from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from alphaomega_reporter.config import ReportConfig
from alphaomega_reporter.viz import plot_lfp_heatmap


def test_lfp_heatmap_frequency_ticks_use_integer_labels() -> None:
    fig, ax = plt.subplots()
    try:
        depths_mm = [-2.0, -1.0, 0.0]
        freq_hz = np.array([0.0, 24.9, 49.8, 74.7, 99.6], dtype=np.float64)
        psd_db_by_depth = np.arange(15, dtype=np.float64).reshape(3, 5)

        plot_lfp_heatmap(
            ax,
            depths_mm,
            freq_hz,
            psd_db_by_depth,
            band_limits_hz=(13.0, 30.0),
            band_name="beta",
            config=ReportConfig(),
        )

        tick_labels = [tick.get_text() for tick in ax.get_yticklabels() if tick.get_text()]

        assert tick_labels
        assert all(label.isdigit() for label in tick_labels)
        assert tick_labels[-1] == "100"
    finally:
        plt.close(fig)
