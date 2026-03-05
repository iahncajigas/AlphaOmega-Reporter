from __future__ import annotations

import numpy as np


def isi_violation_ratio(times_s: np.ndarray, refractory_ms: float) -> float:
    spikes = np.sort(np.asarray(times_s, dtype=np.float64))
    if spikes.size < 2:
        return float("nan")
    isi = np.diff(spikes)
    return float(np.mean(isi < (float(refractory_ms) / 1000.0)))


def presence_ratio(times_s: np.ndarray, start_s: float, stop_s: float, bins: int) -> float:
    spikes = np.asarray(times_s, dtype=np.float64)
    if bins <= 0 or stop_s <= start_s:
        return float("nan")
    edges = np.linspace(float(start_s), float(stop_s), int(bins) + 1)
    counts, _ = np.histogram(spikes, bins=edges)
    return float(np.mean(counts > 0))


def amplitude_cutoff_proxy(p2p_values: np.ndarray) -> float:
    amps = np.asarray(p2p_values, dtype=np.float64)
    amps = amps[np.isfinite(amps)]
    if amps.size < 8:
        return float("nan")
    hist, edges = np.histogram(amps, bins="fd")
    if hist.size < 3 or np.max(hist) <= 0:
        return float("nan")
    mode_idx = int(np.argmax(hist))
    left = hist[: mode_idx + 1]
    right = hist[mode_idx:]
    mirrored = np.pad(right, (0, max(0, left.size - right.size)), mode="constant")[: left.size]
    missing = np.maximum(mirrored - left[::-1], 0)
    return float(np.sum(missing) / max(np.sum(hist), 1))
