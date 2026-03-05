from __future__ import annotations

import numpy as np

from alphaomega_reporter.config import ReportConfig
from alphaomega_reporter.features import (
    band_limits,
    clamp_nperseg,
    compute_bandpower_db,
    compute_welch_psd,
    highpass_filter,
    interpolate_psd_to_grid,
)


def test_beta_bandpower_exceeds_gamma_for_beta_peak() -> None:
    config = ReportConfig()
    fs_hz = 1000.0
    times = np.arange(int(4 * fs_hz), dtype=np.float64) / fs_hz
    values = np.sin(2.0 * np.pi * 20.0 * times) + 0.15 * np.sin(2.0 * np.pi * 45.0 * times)

    psd = compute_welch_psd(
        values,
        fs_hz,
        fmin_hz=config.lfp.fmin_hz,
        fmax_hz=config.lfp.fmax_hz,
        nperseg=config.lfp.welch_nperseg,
        noverlap=config.lfp.welch_noverlap,
    )

    beta = compute_bandpower_db(psd.freq_hz, psd.psd, *band_limits("beta", config))
    gamma = compute_bandpower_db(psd.freq_hz, psd.psd, *band_limits("gamma", config))

    assert psd.freq_hz[0] >= 0.0
    assert psd.freq_hz[-1] <= 100.0
    assert beta > gamma


def test_gamma_bandpower_exceeds_highbeta_and_interpolates_to_reference_grid() -> None:
    config = ReportConfig()
    fs_hz = 1000.0
    t1 = np.arange(int(3 * fs_hz), dtype=np.float64) / fs_hz
    t2 = np.arange(int(2 * fs_hz), dtype=np.float64) / fs_hz
    sig1 = np.sin(2.0 * np.pi * 45.0 * t1)
    sig2 = np.sin(2.0 * np.pi * 45.0 * t2)

    psd1 = compute_welch_psd(
        sig1,
        fs_hz,
        fmin_hz=config.lfp.fmin_hz,
        fmax_hz=config.lfp.fmax_hz,
        nperseg=config.lfp.welch_nperseg,
        noverlap=config.lfp.welch_noverlap,
    )
    psd2 = compute_welch_psd(
        sig2,
        fs_hz,
        fmin_hz=config.lfp.fmin_hz,
        fmax_hz=config.lfp.fmax_hz,
        nperseg=config.lfp.welch_nperseg,
        noverlap=config.lfp.welch_noverlap,
    )
    interp = interpolate_psd_to_grid(psd2.freq_hz, psd2.psd, psd1.freq_hz)
    gamma = compute_bandpower_db(psd1.freq_hz, psd1.psd, *band_limits("gamma", config))
    highbeta = compute_bandpower_db(psd1.freq_hz, psd1.psd, *band_limits("highbeta", config))

    assert interp.shape == psd1.freq_hz.shape
    assert gamma > highbeta


def test_short_trace_helpers_handle_real_validation_edge_cases() -> None:
    short = np.arange(90, dtype=np.float64)

    assert clamp_nperseg(short.size, 512) == 90
    filtered = highpass_filter(short, fs_hz=24000.0, cutoff_hz=300.0)
    assert filtered.shape == short.shape
