from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .config import ReportConfig
from .loaders.map_native import MAPSessionReader
from .loaders.mpx_neo import MPXReaderUnavailableError, MPXSessionReader
from .model import CaseMetadata, ContinuousStream, Segment, SelectedStream, Session, TrajectoryGroup
from .xmlmeta import parse_case_metadata


class AlphaOmegaReporterError(RuntimeError):
    pass


def _iter_root_files(case_dir: Path) -> list[Path]:
    return [path for path in case_dir.iterdir() if path.is_file()]


def _should_ignore(path: Path, config: ReportConfig) -> bool:
    name = path.name
    if name in config.case_metadata.ignore_subdirs:
        return True
    if path.suffix in config.case_metadata.ignore_suffixes:
        return True
    return False


def _detect_case_kind(case_dir: Path, config: ReportConfig) -> tuple[str, list[Path]]:
    files = [path for path in _iter_root_files(case_dir) if not _should_ignore(path, config)]
    map_files = sorted(path for path in files if path.suffix.lower() == ".map")
    if map_files:
        return ("map", map_files)
    mpx_files = sorted(path for path in files if path.suffix.lower() in {".mpx", ".mlx"})
    if mpx_files:
        return ("mpx", mpx_files)
    list_files = sorted(path for path in files if path.suffix.lower() in {".lsm", ".lsx"})
    if len(list_files) == 1:
        return ("list", list_files)
    raise AlphaOmegaReporterError(
        f"Could not determine supported AlphaOmega input files in '{case_dir}'. "
        f"Found map={len(map_files)} mpx={len(mpx_files)} listfiles={len(list_files)}."
    )


def _apply_regex_meta(segment: Segment, source_text: str, pattern: str) -> None:
    match = re.search(pattern, source_text)
    if not match:
        return
    groups = match.groupdict()
    if segment.meta.get("hemisphere") is None and groups.get("hemi"):
        segment.meta["hemisphere"] = groups["hemi"].upper()
    if segment.meta.get("trajectory_number") is None and groups.get("traj"):
        segment.meta["trajectory_number"] = int(groups["traj"])
    if segment.meta.get("depth_mm") is None and groups.get("depth"):
        segment.meta["depth_mm"] = float(groups["depth"])


def normalize_segment_metadata(session: Session, config: ReportConfig) -> None:
    for segment in session.segments:
        if "trajectory_number" in segment.meta:
            try:
                segment.meta["trajectory_number"] = int(segment.meta["trajectory_number"])
            except (TypeError, ValueError):
                segment.meta["trajectory_number"] = None
        else:
            segment.meta["trajectory_number"] = None

        if "depth_mm" in segment.meta:
            try:
                segment.meta["depth_mm"] = float(segment.meta["depth_mm"])
            except (TypeError, ValueError):
                segment.meta["depth_mm"] = None
        else:
            segment.meta["depth_mm"] = None

        if "hemisphere" in segment.meta and segment.meta["hemisphere"] is not None:
            segment.meta["hemisphere"] = str(segment.meta["hemisphere"]).upper()
        else:
            segment.meta["hemisphere"] = None

        for rule in config.depth_parsing.regexes:
            if rule.source == "meta":
                if segment.meta.get("depth_mm") is None:
                    value = segment.meta.get(rule.key or "")
                    try:
                        segment.meta["depth_mm"] = float(value)
                    except (TypeError, ValueError):
                        pass
                continue
            if rule.source == "segment_id" and rule.pattern:
                _apply_regex_meta(segment, segment.segment_id, rule.pattern)
            if rule.source == "source_file" and rule.pattern:
                source_file = str(segment.meta.get("source_file", ""))
                _apply_regex_meta(segment, Path(source_file).name, rule.pattern)

    if not any(segment.meta.get("depth_mm") is not None for segment in session.segments):
        raise AlphaOmegaReporterError("no segments with depth metadata found")


def load_case(case_dir: Path | str, config: ReportConfig) -> Session:
    path = Path(case_dir).expanduser().resolve()
    if not path.is_dir():
        raise AlphaOmegaReporterError(f"case-dir must be a directory: {path}")
    kind, selected_files = _detect_case_kind(path, config)
    if kind == "map":
        session = MAPSessionReader(path).read(scale_to_uv=False, concat_depth_files=True)
    elif kind == "mpx":
        session = MPXSessionReader(path).read(
            include_spikes=True, include_waveforms=True, concat_depth_files=True
        )
    else:
        list_path = selected_files[0]
        if list_path.suffix.lower() == ".lsm":
            session = MAPSessionReader(list_path).read(scale_to_uv=False, concat_depth_files=True)
        else:
            try:
                session = MPXSessionReader(list_path).read(
                    include_spikes=True,
                    include_waveforms=True,
                    concat_depth_files=True,
                )
            except MPXReaderUnavailableError as exc:
                raise AlphaOmegaReporterError(str(exc)) from exc
    normalize_segment_metadata(session, config)
    case_meta = parse_case_metadata(path, config)
    session.meta["case_dir"] = str(path)
    session.meta["case_name"] = path.name
    session.meta["case_metadata"] = {
        "xml_path": case_meta.xml_path,
        "target_case_default": case_meta.target_case_default,
        "target_by_side": case_meta.target_by_side,
        "notes": case_meta.notes,
    }
    return session


def resolve_case_metadata(case_dir: Path | str, config: ReportConfig) -> CaseMetadata:
    return parse_case_metadata(Path(case_dir), config)


def resolve_target(
    side: str, trajectory: int, case_metadata: CaseMetadata, config: ReportConfig
) -> str | None:
    for binding in config.targets.bindings:
        if binding.side == side and binding.trajectory == trajectory:
            return binding.target
    if config.targets.case_default_target:
        return config.targets.case_default_target
    if side in case_metadata.target_by_side:
        return case_metadata.target_by_side[side]
    return case_metadata.target_case_default


def select_stream(
    segment: Segment, hints: Iterable[str], prefer_highest_fs: bool, config: ReportConfig
) -> SelectedStream | None:
    candidates: list[tuple[int, float, str, str, ContinuousStream]] = []
    hint_list = [hint.upper() for hint in hints]
    for key, stream in segment.streams.items():
        haystack = " ".join(
            [
                key,
                stream.stream_id,
                *stream.channel_ids,
                *stream.channel_names,
            ]
        ).upper()
        rank = len(hint_list) + 1
        for idx, hint in enumerate(hint_list):
            if hint in haystack:
                rank = idx
                break
        fs_sort = float(stream.fs_hz)
        if not prefer_highest_fs:
            fs_sort = -fs_sort
        candidates.append((rank, -fs_sort, key, haystack, stream))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _rank, _fs_sort, key, _haystack, stream = candidates[0]
    channel_index = 0
    if (
        stream.data.ndim == 2
        and stream.data.shape[1] > 1
        and config.stream_selection.prefer_channel_with_max_rms
    ):
        data = np.asarray(stream.data, dtype=np.float64)
        rms = np.sqrt(np.nanmean(np.square(data), axis=0))
        if np.any(np.isfinite(rms)):
            channel_index = int(np.nanargmax(rms))
    channel_id = (
        stream.channel_ids[channel_index]
        if channel_index < len(stream.channel_ids)
        else str(channel_index)
    )
    channel_name = (
        stream.channel_names[channel_index]
        if channel_index < len(stream.channel_names)
        else channel_id
    )
    return SelectedStream(
        stream_key=key,
        stream_id=stream.stream_id,
        channel_index=channel_index,
        channel_id=channel_id,
        channel_name=channel_name,
        fs_hz=float(stream.fs_hz),
        units=stream.units,
    )


def select_mer_stream(segment: Segment, config: ReportConfig) -> SelectedStream | None:
    return select_stream(
        segment, config.stream_selection.mer_name_hints, prefer_highest_fs=True, config=config
    )


def select_lfp_stream(segment: Segment, config: ReportConfig) -> SelectedStream | None:
    return select_stream(
        segment, config.stream_selection.lfp_name_hints, prefer_highest_fs=False, config=config
    )


def build_trajectory_groups(session: Session, config: ReportConfig) -> list[TrajectoryGroup]:
    case_dir = Path(session.meta.get("case_dir", session.session_path))
    case_metadata = resolve_case_metadata(case_dir, config)
    grouped: dict[tuple[str, int], list[Segment]] = defaultdict(list)
    for segment in session.segments:
        side = segment.meta.get("hemisphere")
        traj = segment.meta.get("trajectory_number")
        depth = segment.meta.get("depth_mm")
        if side is None or traj is None or depth is None:
            continue
        grouped[(str(side), int(traj))].append(segment)

    out: list[TrajectoryGroup] = []
    for (side, trajectory), segments in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        ordered_segments = sorted(segments, key=lambda segment: float(segment.meta["depth_mm"]))
        target = resolve_target(side, trajectory, case_metadata, config)
        if config.render.group_by == "target" and target is None:
            raise AlphaOmegaReporterError(
                f"group-by target requested but no target resolved for {side}T{trajectory}"
            )
        out.append(
            TrajectoryGroup(
                side=side,
                trajectory=trajectory,
                target=target,
                segments=ordered_segments,
                depth_values=[float(segment.meta["depth_mm"]) for segment in ordered_segments],
                warnings=[],
            )
        )
    return out
