from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import ReportConfig
from .io import load_case
from .model import EventSeries, Segment, Session, SpikeStream


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _to_json(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False)


def _safe_name(name: str) -> str:
    return name.replace("/", "_")


def write_session_h5(session: Session, out_path: Path | str) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        meta_group = h5.create_group("meta")
        meta_group.attrs["schema_version"] = "alphaomega_reporter_ao_h5_v1"
        meta_group.attrs["created_utc_iso"] = datetime.now(timezone.utc).isoformat()
        meta_group.create_dataset("session_json", data=_to_json(session.meta), dtype=str_dtype)

        segments_group = h5.create_group("segments")
        for seg_idx, segment in enumerate(session.segments):
            _write_segment(segments_group.create_group(f"seg{seg_idx:04d}"), segment, str_dtype)
    return path


def export_h5(
    case_dir: Path | str, out_path: Path | str, config: ReportConfig | None = None
) -> Path:
    session = load_case(case_dir, config or ReportConfig())
    return write_session_h5(session, out_path)


def _write_segment(group: h5py.Group, segment: Segment, str_dtype: Any) -> None:
    group.attrs["segment_id"] = segment.segment_id
    group.create_dataset("meta_json", data=_to_json(segment.meta), dtype=str_dtype)

    streams_group = group.create_group("streams")
    for stream_key, stream in segment.streams.items():
        stream_group = streams_group.create_group(_safe_name(stream_key))
        stream_group.attrs["stream_key"] = stream_key
        stream_group.attrs["stream_id"] = stream.stream_id
        stream_group.create_dataset("fs_hz", data=float(stream.fs_hz))
        stream_group.create_dataset("units", data=stream.units, dtype=str_dtype)
        stream_group.create_dataset("t0_s", data=float(stream.t0_s))
        stream_group.create_dataset(
            "channel_ids", data=np.asarray(stream.channel_ids, dtype=object), dtype=str_dtype
        )
        stream_group.create_dataset(
            "channel_names",
            data=np.asarray(stream.channel_names, dtype=object),
            dtype=str_dtype,
        )
        stream_group.create_dataset("source_json", data=_to_json(stream.source), dtype=str_dtype)
        stream_group.create_dataset(
            "data", data=np.asarray(stream.data, dtype=np.float32), compression="gzip"
        )

    spikes_group = group.create_group("spikes")
    for spike_key, spike in segment.spikes.items():
        _write_spike(spikes_group.create_group(_safe_name(spike_key)), spike_key, spike, str_dtype)

    events_group = group.create_group("events")
    for event_key, event in segment.events.items():
        _write_event(events_group.create_group(_safe_name(event_key)), event_key, event, str_dtype)


def _write_spike(group: h5py.Group, spike_key: str, spike: SpikeStream, str_dtype: Any) -> None:
    group.attrs["spike_key"] = spike_key
    group.attrs["spike_id"] = spike.spike_id
    group.create_dataset("fs_hz", data=float(spike.fs_hz))
    group.create_dataset("times_s", data=np.asarray(spike.times_s, dtype=np.float64))
    group.create_dataset("source_json", data=_to_json(spike.source), dtype=str_dtype)
    if spike.waveforms is not None:
        group.create_dataset(
            "waveforms", data=np.asarray(spike.waveforms, dtype=np.float32), compression="gzip"
        )
    if spike.unit_ids is not None:
        group.create_dataset("unit_ids", data=np.asarray(spike.unit_ids))
    if spike.pre_s is not None:
        group.create_dataset("pre_s", data=float(spike.pre_s))
    if spike.post_s is not None:
        group.create_dataset("post_s", data=float(spike.post_s))


def _write_event(group: h5py.Group, event_key: str, event: EventSeries, str_dtype: Any) -> None:
    group.attrs["event_key"] = event_key
    group.attrs["event_id"] = event.event_id
    group.create_dataset("times_s", data=np.asarray(event.times_s, dtype=np.float64))
    group.create_dataset("source_json", data=_to_json(event.source), dtype=str_dtype)
    if event.values is None:
        return
    if isinstance(event.values, list):
        group.create_dataset("values", data=np.asarray(event.values, dtype=object), dtype=str_dtype)
        group.attrs["values_kind"] = "str"
        return
    values = np.asarray(event.values)
    if values.dtype.kind in {"U", "S", "O"}:
        group.create_dataset("values", data=values.astype(str), dtype=str_dtype)
        group.attrs["values_kind"] = "str"
    else:
        group.create_dataset("values", data=values)
        group.attrs["values_kind"] = "numeric"
