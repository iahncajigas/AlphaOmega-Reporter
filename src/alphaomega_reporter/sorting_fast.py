from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .config import ReportConfig
from .features import highpass_filter
from .model import UnitSummary
from .qc import amplitude_cutoff_proxy, isi_violation_ratio, presence_ratio


@dataclass
class FastSortingResult:
    spike_times_s: np.ndarray
    labels: np.ndarray
    waveforms: np.ndarray
    unit_summaries: list[UnitSummary]
    warnings: list[str]


def _detect_spikes(
    values: np.ndarray, fs_hz: float, threshold_mad: float, refractory_ms: float
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    mad = np.median(np.abs(data - np.median(data))) + np.finfo(float).eps
    threshold = -float(threshold_mad) * mad / 0.6745
    candidate_idx = np.flatnonzero(data < threshold)
    if candidate_idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    refractory_samples = max(int(round((float(refractory_ms) / 1000.0) * float(fs_hz))), 1)
    kept: list[int] = []
    last = -refractory_samples
    for idx in candidate_idx:
        if idx - last < refractory_samples:
            continue
        local_start = max(idx - 1, 0)
        local_stop = min(idx + 2, data.size)
        local = data[local_start:local_stop]
        kept.append(int(local_start + np.argmin(local)))
        last = kept[-1]
    return np.asarray(kept, dtype=np.int64)


def _extract_waveforms(
    values: np.ndarray, spike_idx: np.ndarray, fs_hz: float, pre_ms: float, post_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    pre = max(int(round(float(pre_ms) * float(fs_hz) / 1000.0)), 1)
    post = max(int(round(float(post_ms) * float(fs_hz) / 1000.0)), 1)
    kept_idx: list[int] = []
    waveforms: list[np.ndarray] = []
    for idx in spike_idx:
        start = int(idx) - pre
        stop = int(idx) + post
        if start < 0 or stop >= data.size:
            continue
        snippet = data[start:stop].astype(np.float32, copy=True)
        if not np.all(np.isfinite(snippet)):
            continue
        kept_idx.append(int(idx))
        waveforms.append(snippet)
    if not waveforms:
        return np.zeros(0, dtype=np.int64), np.zeros((0, pre + post), dtype=np.float32)
    return np.asarray(kept_idx, dtype=np.int64), np.vstack(waveforms)


def _cluster_features(waveforms: np.ndarray, max_clusters: int) -> np.ndarray:
    if waveforms.shape[0] < 4:
        return np.zeros(waveforms.shape[0], dtype=np.int32)
    centered = waveforms - np.mean(waveforms, axis=1, keepdims=True)
    n_components = min(3, centered.shape[0], centered.shape[1])
    features = PCA(n_components=n_components).fit_transform(centered)

    try:
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, min(20, waveforms.shape[0] // 4)))
        labels = clusterer.fit_predict(features)
        if np.any(labels >= 0):
            unique = np.unique(labels[labels >= 0])
            remap = {label: idx for idx, label in enumerate(unique)}
            return np.asarray([remap.get(int(label), -1) for label in labels], dtype=np.int32)
    except Exception:
        pass

    best_score = -np.inf
    best_labels = np.zeros(waveforms.shape[0], dtype=np.int32)
    upper = min(int(max_clusters), waveforms.shape[0] - 1)
    for n_clusters in range(2, upper + 1):
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = model.fit_predict(features)
        score = silhouette_score(features, labels) if len(np.unique(labels)) > 1 else -np.inf
        if score > best_score:
            best_score = score
            best_labels = labels.astype(np.int32)
    return best_labels


def sort_single_channel(
    values: np.ndarray,
    fs_hz: float,
    depth_mm: float,
    config: ReportConfig,
) -> FastSortingResult:
    filtered = highpass_filter(values, fs_hz, config.sorting.highpass_hz)
    spike_idx = _detect_spikes(
        filtered,
        fs_hz,
        threshold_mad=config.sorting.threshold_mad,
        refractory_ms=config.qc.refractory_ms,
    )
    spike_idx, waveforms = _extract_waveforms(
        filtered,
        spike_idx,
        fs_hz,
        pre_ms=config.sorting.pre_ms,
        post_ms=config.sorting.post_ms,
    )
    if spike_idx.size == 0 or waveforms.size == 0:
        return FastSortingResult(
            spike_times_s=np.zeros(0, dtype=np.float64),
            labels=np.zeros(0, dtype=np.int32),
            waveforms=np.zeros((0, 0), dtype=np.float32),
            unit_summaries=[],
            warnings=["No spikes detected"],
        )
    labels = _cluster_features(waveforms, config.sorting.max_kmeans_clusters)
    if np.all(labels < 0):
        labels = np.zeros(waveforms.shape[0], dtype=np.int32)
    unlabeled = labels < 0
    if np.any(unlabeled):
        labels[unlabeled] = int(np.max(labels[~unlabeled]) + 1 if np.any(~unlabeled) else 0)
    spike_times_s = spike_idx.astype(np.float64) / float(fs_hz)
    duration_s = (
        float(max(spike_times_s[-1], waveforms.shape[0] / fs_hz))
        if spike_times_s.size
        else float(len(values) / fs_hz)
    )
    unit_summaries: list[UnitSummary] = []
    for unit_id in sorted(np.unique(labels)):
        mask = labels == unit_id
        unit_times = spike_times_s[mask]
        unit_waveforms = waveforms[mask]
        p2p = (
            np.ptp(unit_waveforms, axis=1) if unit_waveforms.size else np.zeros(0, dtype=np.float32)
        )
        unit_summaries.append(
            UnitSummary(
                depth_mm=float(depth_mm),
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
    return FastSortingResult(
        spike_times_s=spike_times_s,
        labels=labels,
        waveforms=waveforms,
        unit_summaries=unit_summaries,
        warnings=[],
    )
