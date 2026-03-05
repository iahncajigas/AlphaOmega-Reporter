from __future__ import annotations

from tempfile import TemporaryDirectory

import numpy as np

from .config import ReportConfig
from .model import UnitSummary
from .qc import amplitude_cutoff_proxy, isi_violation_ratio, presence_ratio
from .sorting_fast import FastSortingResult, sort_single_channel


def sort_with_spikeinterface(
    values, fs_hz: float, depth_mm: float, config: ReportConfig
) -> FastSortingResult:
    try:
        import spikeinterface.core as sicore
        import spikeinterface.preprocessing as sipre
        import spikeinterface.sorters as sisorters
    except Exception:
        result = sort_single_channel(values, fs_hz, depth_mm, config)
        result.warnings.append("SpikeInterface unavailable; fell back to fast sorter")
        return result

    recording = sicore.NumpyRecording(
        traces_list=[np.asarray(values, dtype=np.float32)[:, None]],
        sampling_frequency=float(fs_hz),
    )
    recording.set_dummy_probe_from_locations(np.asarray([[0.0, 0.0]], dtype=np.float32))
    recording = sipre.bandpass_filter(
        recording,
        freq_min=300.0,
        freq_max=min(6000.0, fs_hz / 2.5),
    )
    if config.sorting.common_reference:
        try:
            recording = sipre.common_reference(recording, reference="global")
        except Exception:
            pass

    sorter_name = config.sorting.spikeinterface_sorter
    sorter_class = sisorters.sorter_dict.get(sorter_name)
    if sorter_class is None or not sorter_class.is_installed():
        result = sort_single_channel(values, fs_hz, depth_mm, config)
        result.warnings.append(
            f"SpikeInterface sorter '{sorter_name}' unavailable; fell back to fast sorter"
        )
        return result

    try:
        with TemporaryDirectory(prefix="ao_report_si_") as folder:
            recording_folder = f"{folder}/recording"
            sorter_folder = f"{folder}/sorter"
            recording_saved = recording.save(
                folder=recording_folder,
                overwrite=True,
                verbose=False,
            )
            sorting = sisorters.run_sorter(
                sorter_name=sorter_name,
                recording=recording_saved,
                folder=sorter_folder,
                remove_existing_folder=True,
                verbose=False,
                raise_error=True,
            )
    except Exception as exc:
        result = sort_single_channel(values, fs_hz, depth_mm, config)
        result.warnings.append(
            "SpikeInterface sorter "
            f"'{sorter_name}' failed ({_short_error(exc)}); fell back to fast sorter"
        )
        return result

    try:
        result = _summarize_sorting(values, fs_hz, depth_mm, sorting, config)
        result.warnings.append(f"SpikeInterface sorter '{sorter_name}' completed")
        return result
    except Exception as exc:
        result = sort_single_channel(values, fs_hz, depth_mm, config)
        result.warnings.append(
            "SpikeInterface result conversion failed "
            f"({_short_error(exc)}); fell back to fast sorter"
        )
        return result


def _summarize_sorting(
    values: np.ndarray,
    fs_hz: float,
    depth_mm: float,
    sorting,
    config: ReportConfig,
) -> FastSortingResult:
    trains: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label, unit_id in enumerate(sorting.get_unit_ids()):
        spike_idx = np.asarray(sorting.get_unit_spike_train(unit_id), dtype=np.int64)
        spike_idx = spike_idx[(spike_idx >= 0) & (spike_idx < len(values))]
        if spike_idx.size == 0:
            continue
        trains.append(spike_idx)
        labels.append(np.full(spike_idx.shape, label, dtype=np.int32))
    if not trains:
        return FastSortingResult(
            spike_times_s=np.zeros(0, dtype=np.float64),
            labels=np.zeros(0, dtype=np.int32),
            waveforms=np.zeros((0, 0), dtype=np.float32),
            unit_summaries=[],
            warnings=["SpikeInterface sorter returned no spikes"],
        )

    spike_idx = np.concatenate(trains)
    unit_labels = np.concatenate(labels)
    order = np.argsort(spike_idx, kind="mergesort")
    spike_idx = spike_idx[order]
    unit_labels = unit_labels[order]

    spike_idx, unit_labels, waveforms = _extract_labeled_waveforms(
        values,
        spike_idx,
        unit_labels,
        fs_hz,
        config.sorting.pre_ms,
        config.sorting.post_ms,
    )
    if spike_idx.size == 0:
        return FastSortingResult(
            spike_times_s=np.zeros(0, dtype=np.float64),
            labels=np.zeros(0, dtype=np.int32),
            waveforms=np.zeros((0, 0), dtype=np.float32),
            unit_summaries=[],
            warnings=["SpikeInterface spikes were all dropped during waveform extraction"],
        )

    spike_times_s = spike_idx.astype(np.float64) / float(fs_hz)
    duration_s = max(
        float(len(values) / fs_hz),
        float(spike_times_s[-1]) if spike_times_s.size else 0.0,
    )
    unit_summaries: list[UnitSummary] = []
    for unit_id in sorted(np.unique(unit_labels)):
        mask = unit_labels == unit_id
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
                    unit_times,
                    0.0,
                    duration_s,
                    config.qc.presence_ratio_bins,
                ),
                amplitude_cutoff_proxy=amplitude_cutoff_proxy(p2p),
            )
        )
    return FastSortingResult(
        spike_times_s=spike_times_s,
        labels=unit_labels,
        waveforms=waveforms,
        unit_summaries=unit_summaries,
        warnings=[],
    )


def _extract_labeled_waveforms(
    values: np.ndarray,
    spike_idx: np.ndarray,
    labels: np.ndarray,
    fs_hz: float,
    pre_ms: float,
    post_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    pre = max(int(round(float(pre_ms) * float(fs_hz) / 1000.0)), 1)
    post = max(int(round(float(post_ms) * float(fs_hz) / 1000.0)), 1)
    kept_idx: list[int] = []
    kept_labels: list[int] = []
    waveforms: list[np.ndarray] = []
    for idx, label in zip(spike_idx, labels, strict=False):
        start = int(idx) - pre
        stop = int(idx) + post
        if start < 0 or stop >= data.size:
            continue
        snippet = data[start:stop].astype(np.float32, copy=True)
        if not np.all(np.isfinite(snippet)):
            continue
        kept_idx.append(int(idx))
        kept_labels.append(int(label))
        waveforms.append(snippet)
    if not waveforms:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, pre + post), dtype=np.float32),
        )
    return (
        np.asarray(kept_idx, dtype=np.int64),
        np.asarray(kept_labels, dtype=np.int32),
        np.vstack(waveforms),
    )


def _short_error(exc: Exception) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    return text[:160]
