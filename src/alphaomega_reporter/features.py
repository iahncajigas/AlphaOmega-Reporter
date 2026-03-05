from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import ReportConfig


@dataclass
class PSDResult:
    freq_hz: np.ndarray
    psd: np.ndarray
    psd_db: np.ndarray


def ordered_bands(primary_band: str, plot_bands: Iterable[str], config: ReportConfig) -> list[str]:
    canonical = ["delta", "theta", "alpha", "beta", "highbeta", "gamma"]
    requested = [primary_band, *list(plot_bands)]
    seen: set[str] = set()
    requested = [band for band in requested if not (band in seen or seen.add(band))]
    extra = [band for band in requested if band not in canonical]
    ordered = [band for band in canonical if band in requested]
    ordered.extend(extra)
    return ordered


def band_limits(band_name: str, config: ReportConfig) -> tuple[float, float]:
    if band_name not in config.bands.definitions:
        raise ValueError(f"Unknown band '{band_name}'")
    band = config.bands.definitions[band_name]
    return float(band.low_hz), float(band.high_hz)


def _as_1d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[:, 0]
    return array.reshape(-1)


def finite_signal(values: np.ndarray) -> np.ndarray:
    array = _as_1d(values)
    return array[np.isfinite(array)]


def clamp_nperseg(n_samples: int, requested: int) -> int:
    if n_samples < 8:
        raise ValueError("signal too short for PSD")
    if n_samples <= 128:
        return int(n_samples)
    out = min(int(requested), int(n_samples))
    power = 1
    while power * 2 <= out:
        power *= 2
    return max(128, power)


def compute_welch_psd(
    values: np.ndarray,
    fs_hz: float,
    *,
    fmin_hz: float,
    fmax_hz: float,
    nperseg: int,
    noverlap: int,
) -> PSDResult:
    data = finite_signal(values)
    if data.size < 8:
        raise ValueError("signal too short for PSD")
    nperseg_eff = clamp_nperseg(data.size, nperseg)
    noverlap_eff = min(int(noverlap), max(nperseg_eff - 1, 0))
    freq_hz, psd = signal.welch(data, fs=float(fs_hz), nperseg=nperseg_eff, noverlap=noverlap_eff)
    mask = (freq_hz >= float(fmin_hz)) & (freq_hz <= float(fmax_hz))
    freq_hz = np.asarray(freq_hz[mask], dtype=np.float64)
    psd = np.asarray(psd[mask], dtype=np.float64)
    return PSDResult(freq_hz=freq_hz, psd=psd, psd_db=10.0 * np.log10(psd + np.finfo(float).eps))


def compute_multitaper_psd(
    values: np.ndarray,
    fs_hz: float,
    *,
    fmin_hz: float,
    fmax_hz: float,
    nw: float = 4.0,
) -> PSDResult:
    data = finite_signal(values)
    if data.size < 8:
        raise ValueError("signal too short for PSD")
    n = int(data.size)
    nfft = max(256, 1 << (n - 1).bit_length())
    kmax = max(int(2.0 * nw - 1.0), 1)
    tapers = signal.windows.dpss(n, nw, Kmax=kmax, sym=False)
    spec = np.zeros(nfft // 2 + 1, dtype=np.float64)
    for taper in tapers:
        tapered = data * taper
        fft_vals = np.fft.rfft(tapered, n=nfft)
        power = (np.abs(fft_vals) ** 2) / (float(fs_hz) * float(np.sum(taper**2)))
        if nfft % 2 == 0:
            power[1:-1] *= 2.0
        else:
            power[1:] *= 2.0
        spec += power
    spec /= float(kmax)
    freq_hz = np.fft.rfftfreq(nfft, d=1.0 / float(fs_hz))
    mask = (freq_hz >= float(fmin_hz)) & (freq_hz <= float(fmax_hz))
    freq_hz = np.asarray(freq_hz[mask], dtype=np.float64)
    psd = np.asarray(spec[mask], dtype=np.float64)
    return PSDResult(freq_hz=freq_hz, psd=psd, psd_db=10.0 * np.log10(psd + np.finfo(float).eps))


def interpolate_psd_to_grid(
    freq_hz: np.ndarray, psd: np.ndarray, ref_freq_hz: np.ndarray
) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    ref = np.asarray(ref_freq_hz, dtype=np.float64)
    if freq_hz.shape == ref.shape and np.allclose(freq_hz, ref):
        return psd
    freq_unique, unique_idx = np.unique(freq_hz, return_index=True)
    psd_unique = psd[unique_idx]
    return np.interp(ref, freq_unique, psd_unique, left=np.nan, right=np.nan)


def integrate_bandpower(
    freq_hz: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float
) -> float:
    mask = (freq_hz >= float(low_hz)) & (freq_hz <= float(high_hz))
    if np.count_nonzero(mask) < 2:
        return float("nan")
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(psd[mask], freq_hz[mask]))
    return float(np.trapz(psd[mask], freq_hz[mask]))


def compute_bandpower_db(
    freq_hz: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float
) -> float:
    power = integrate_bandpower(freq_hz, psd, low_hz, high_hz)
    if not np.isfinite(power):
        return float("nan")
    return float(10.0 * np.log10(power + np.finfo(float).eps))


def compute_mer_rms(values: np.ndarray) -> float:
    data = finite_signal(values)
    if data.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(data))))


def highpass_filter(values: np.ndarray, fs_hz: float, cutoff_hz: float) -> np.ndarray:
    data = _as_1d(values)
    if cutoff_hz <= 0 or fs_hz <= cutoff_hz * 2:
        return data.astype(np.float64, copy=True)
    b, a = signal.butter(3, float(cutoff_hz), btype="highpass", fs=float(fs_hz))
    padlen = 3 * (max(len(a), len(b)) - 1)
    if data.size <= padlen:
        return data.astype(np.float64, copy=True)
    return signal.filtfilt(b, a, data).astype(np.float64, copy=False)


def compute_mua_envelope(values: np.ndarray, fs_hz: float, highpass_hz: float) -> np.ndarray:
    filtered = highpass_filter(values, fs_hz, highpass_hz)
    rectified = np.abs(filtered)
    win = max(int(round(float(fs_hz) * 0.010)), 1)
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(rectified, kernel, mode="same")
