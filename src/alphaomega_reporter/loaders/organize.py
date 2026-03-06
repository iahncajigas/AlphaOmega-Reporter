from __future__ import annotations

from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any

import numpy as np

from ..model import ContinuousStream, EventSeries, Segment, SpikeStream


def _group_key(segment: Segment) -> tuple[Any, int, float]:
    hemi = segment.meta.get("hemisphere")
    traj = int(segment.meta.get("trajectory_number", 0))
    depth = round(float(segment.meta.get("depth_mm", 0.0)), 6)
    return (hemi, traj, depth)


def _segment_sort_key(segment: Segment) -> tuple[int, str, str]:
    file_index = int(segment.meta.get("file_index", 0))
    tag = str(segment.meta.get("tag", ""))
    return (file_index, tag, segment.segment_id)


def _meta_float(meta: dict[str, Any], key: str) -> float | None:
    value = meta.get(key)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _segment_duration_s(segment: Segment) -> float:
    duration = 0.0
    for stream in segment.streams.values():
        if stream.fs_hz > 0:
            duration = max(
                duration, float(stream.t0_s) + float(stream.data.shape[0]) / float(stream.fs_hz)
            )
    for spike in segment.spikes.values():
        if spike.times_s.size:
            duration = max(duration, float(np.max(spike.times_s)))
    for event in segment.events.values():
        if event.times_s.size:
            duration = max(duration, float(np.max(event.times_s)))
    return duration


def _compatible_stream(a: ContinuousStream, b: ContinuousStream) -> bool:
    return (
        np.isclose(a.fs_hz, b.fs_hz)
        and a.units == b.units
        and a.channel_ids == b.channel_ids
        and a.channel_names == b.channel_names
    )


def _merge_streams(
    segments: list[Segment],
    offsets_s: list[float],
    source_files: list[str],
) -> tuple[dict[str, ContinuousStream], list[str]]:
    merged: dict[str, ContinuousStream] = OrderedDict()
    warnings: list[str] = []

    keys: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        for key in segment.streams:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    for key in keys:
        availability = [key in segment.streams for segment in segments]
        if not all(availability):
            warnings.append(f"Stream '{key}' missing in some files; kept as per-file parts.")
            for idx, segment in enumerate(segments):
                if key not in segment.streams:
                    continue
                part = deepcopy(segment.streams[key])
                part_key = f"{key}__part{idx + 1:02d}"
                part.source = {
                    **part.source,
                    "concat_part_index": idx,
                    "concat_offset_s": offsets_s[idx],
                    "source_file": source_files[idx],
                }
                merged[part_key] = part
            continue

        streams = [segment.streams[key] for segment in segments]
        if not all(_compatible_stream(streams[0], item) for item in streams[1:]):
            warnings.append(
                f"Stream '{key}' metadata mismatch across files; kept as per-file parts."
            )
            for idx, stream in enumerate(streams):
                part = deepcopy(stream)
                part_key = f"{key}__part{idx + 1:02d}"
                part.source = {
                    **part.source,
                    "concat_part_index": idx,
                    "concat_offset_s": offsets_s[idx],
                    "source_file": source_files[idx],
                }
                merged[part_key] = part
            continue

        base = deepcopy(streams[0])
        fs = float(base.fs_hz)
        if not np.isfinite(fs) or fs <= 0:
            base.data = np.concatenate(
                [np.asarray(stream.data, dtype=np.float32) for stream in streams], axis=0
            )
            base.t0_s = 0.0
            merged[key] = base
            continue

        starts = [
            int(round((offsets_s[idx] + float(stream.t0_s)) * fs))
            for idx, stream in enumerate(streams)
        ]
        min_start = min(starts)
        if min_start < 0:
            starts = [start - min_start for start in starts]

        blocks: list[np.ndarray] = []
        n_channels = 0
        for stream in streams:
            block = np.asarray(stream.data, dtype=np.float32)
            if block.ndim == 1:
                block = block[:, None]
            n_channels = max(n_channels, block.shape[1])
            blocks.append(block)

        total_samples = max(
            start + block.shape[0] for start, block in zip(starts, blocks, strict=False)
        )
        merged_data = np.full((total_samples, n_channels), np.nan, dtype=np.float32)
        for start, block in zip(starts, blocks, strict=False):
            stop = start + block.shape[0]
            merged_data[start:stop, : block.shape[1]] = block

        base.data = merged_data
        base.t0_s = 0.0
        base.source = {
            **base.source,
            "concatenated_files": source_files,
            "concat_offsets_s": offsets_s,
            "concat_count": len(streams),
        }
        merged[key] = base

    return merged, warnings


def _merge_spike_streams(
    segments: list[Segment],
    offsets_s: list[float],
    source_files: list[str],
) -> dict[str, SpikeStream]:
    merged: dict[str, SpikeStream] = OrderedDict()

    keys: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        for key in segment.spikes:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    for key in keys:
        items: list[tuple[int, SpikeStream]] = []
        for idx, segment in enumerate(segments):
            if key in segment.spikes:
                items.append((idx, segment.spikes[key]))
        if not items:
            continue

        ref = items[0][1]
        times_parts = [
            np.asarray(spike.times_s, dtype=np.float64) + offsets_s[idx] for idx, spike in items
        ]
        times = (
            np.concatenate(times_parts, axis=0) if times_parts else np.zeros(0, dtype=np.float64)
        )
        order = np.argsort(times, kind="mergesort") if times.size else np.zeros(0, dtype=np.int64)
        if times.size:
            times = times[order]

        unit_ids = None
        if all(spike.unit_ids is not None for _, spike in items):
            unit_ids_raw = np.concatenate(
                [np.asarray(spike.unit_ids) for _, spike in items], axis=0
            )
            unit_ids = (
                unit_ids_raw[order] if unit_ids_raw.shape[0] == times.shape[0] else unit_ids_raw
            )

        waveforms = None
        if all(spike.waveforms is not None for _, spike in items):
            wf_parts = [np.asarray(spike.waveforms, dtype=np.float32) for _, spike in items]
            max_len = max(wf.shape[1] for wf in wf_parts if wf.ndim == 2)
            padded: list[np.ndarray] = []
            for wf in wf_parts:
                if wf.shape[1] == max_len:
                    padded.append(wf)
                    continue
                out = np.full((wf.shape[0], max_len), np.nan, dtype=np.float32)
                out[:, : wf.shape[1]] = wf
                padded.append(out)
            wf_raw = np.concatenate(padded, axis=0)
            waveforms = wf_raw[order] if wf_raw.shape[0] == times.shape[0] else wf_raw

        merged[key] = SpikeStream(
            spike_id=ref.spike_id,
            fs_hz=float(ref.fs_hz),
            times_s=times,
            unit_ids=unit_ids,
            waveforms=waveforms,
            pre_s=ref.pre_s,
            post_s=ref.post_s,
            source={
                **ref.source,
                "concatenated_files": [source_files[idx] for idx, _ in items],
                "concat_offsets_s": [offsets_s[idx] for idx, _ in items],
                "concat_count": len(items),
            },
        )
    return merged


def _merge_event_streams(
    segments: list[Segment],
    offsets_s: list[float],
    source_files: list[str],
) -> dict[str, EventSeries]:
    merged: dict[str, EventSeries] = OrderedDict()
    keys: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        for key in segment.events:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    for key in keys:
        items: list[tuple[int, EventSeries]] = []
        for idx, segment in enumerate(segments):
            if key in segment.events:
                items.append((idx, segment.events[key]))
        if not items:
            continue
        ref = items[0][1]
        times_parts = [
            np.asarray(event.times_s, dtype=np.float64) + offsets_s[idx] for idx, event in items
        ]
        times = (
            np.concatenate(times_parts, axis=0) if times_parts else np.zeros(0, dtype=np.float64)
        )
        order = np.argsort(times, kind="mergesort") if times.size else np.zeros(0, dtype=np.int64)
        if times.size:
            times = times[order]
        merged[key] = EventSeries(
            event_id=ref.event_id,
            times_s=times,
            values=None,
            source={
                **ref.source,
                "concatenated_files": [source_files[idx] for idx, _ in items],
                "concat_offsets_s": [offsets_s[idx] for idx, _ in items],
                "concat_count": len(items),
            },
        )
    return merged


def _build_segment_id(meta: dict[str, Any]) -> str:
    hemi = meta.get("hemisphere")
    prefix_i = bool(meta.get("prefix_i", False))
    traj = int(meta.get("trajectory_number", 0))
    depth = float(meta.get("depth_mm", 0.0))
    lead = ""
    if hemi:
        lead = str(hemi)
    elif prefix_i:
        lead = "i"
    return f"{lead}T{traj}D{depth:.3f}"


def group_and_concatenate_segments(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []

    groups: dict[tuple[Any, int, float], list[Segment]] = defaultdict(list)
    for segment in segments:
        groups[_group_key(segment)].append(segment)

    grouped_segments: list[Segment] = []
    for key in sorted(groups.keys(), key=lambda item: (str(item[0]), item[1], item[2])):
        unsorted_members = groups[key]
        source_starts_raw = [
            _meta_float(segment.meta, "source_start_s") for segment in unsorted_members
        ]
        have_source_starts = all(item is not None for item in source_starts_raw)
        if have_source_starts:
            members = sorted(
                unsorted_members,
                key=lambda segment: (
                    (float(segment.meta["source_start_s"]),) + _segment_sort_key(segment)
                ),
            )
            sort_basis = "source_start_s"
        else:
            members = sorted(unsorted_members, key=_segment_sort_key)
            sort_basis = "file_index_tag"

        source_files = [
            str(segment.meta.get("source_file", segment.segment_id)) for segment in members
        ]
        durations = [_segment_duration_s(segment) for segment in members]
        offsets_s: list[float] = []
        running = 0.0
        for duration in durations:
            offsets_s.append(running)
            running += float(duration)

        merged_streams, stream_warnings = _merge_streams(members, offsets_s, source_files)
        merged_spikes = _merge_spike_streams(members, offsets_s, source_files)
        merged_events = _merge_event_streams(members, offsets_s, source_files)

        base_meta = deepcopy(members[0].meta)
        base_meta["source_files"] = source_files
        base_meta["source_segment_ids"] = [segment.segment_id for segment in members]
        base_meta["file_indices"] = [int(segment.meta.get("file_index", 0)) for segment in members]
        base_meta["tags"] = [str(segment.meta.get("tag", "")) for segment in members]
        base_meta["concat_offsets_s"] = offsets_s
        base_meta["concat_durations_s"] = durations
        base_meta["concat_boundaries_s"] = [
            offset + duration for offset, duration in zip(offsets_s, durations, strict=False)
        ]
        base_meta["n_concat_files"] = len(members)
        base_meta["grouping"] = ["hemisphere", "trajectory_number", "depth_mm"]
        base_meta["concat_sort_basis"] = sort_basis
        if stream_warnings:
            base_meta["concat_warnings"] = stream_warnings

        grouped_segments.append(
            Segment(
                segment_id=_build_segment_id(base_meta),
                meta=base_meta,
                streams=merged_streams,
                spikes=merged_spikes,
                events=merged_events,
            )
        )
    return grouped_segments
