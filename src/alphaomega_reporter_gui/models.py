from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alphaomega_reporter import ReportConfig
from alphaomega_reporter.io import (
    resolve_case_metadata,
    resolve_target,
    select_lfp_stream,
    select_mer_stream,
)
from alphaomega_reporter.model import ContinuousStream, Segment, SelectedStream, Session


@dataclass
class SegmentModel:
    segment: Segment
    side: str | None
    trajectory: int | None
    target: str | None
    depth_mm: float | None
    mer_selection: SelectedStream | None
    lfp_selection: SelectedStream | None
    warnings: list[str] = field(default_factory=list)

    @property
    def segment_id(self) -> str:
        return self.segment.segment_id

    def _get_stream_and_channel(
        self,
        selection: SelectedStream | None,
        requested_channel: int | None = None,
    ) -> tuple[ContinuousStream | None, int]:
        if selection is None:
            return None, 0
        stream = self.segment.streams.get(selection.stream_key)
        if stream is None:
            return None, 0
        if stream.data.ndim == 1:
            return stream, 0
        channel_index = selection.channel_index if requested_channel is None else requested_channel
        channel_index = max(0, min(int(channel_index), stream.data.shape[1] - 1))
        return stream, channel_index

    def available_mer_channels(self) -> list[str]:
        stream, _ = self._get_stream_and_channel(self.mer_selection)
        if stream is None:
            return []
        if stream.data.ndim == 1:
            return [stream.channel_names[0] if stream.channel_names else "0"]
        return stream.channel_names or [str(idx) for idx in range(stream.data.shape[1])]

    def available_lfp_channels(self) -> list[str]:
        stream, _ = self._get_stream_and_channel(self.lfp_selection)
        if stream is None:
            return []
        if stream.data.ndim == 1:
            return [stream.channel_names[0] if stream.channel_names else "0"]
        return stream.channel_names or [str(idx) for idx in range(stream.data.shape[1])]

    def extract_mer(
        self, channel_index: int | None = None
    ) -> tuple[np.ndarray | None, float | None, str]:
        stream, index = self._get_stream_and_channel(self.mer_selection, channel_index)
        if stream is None:
            return None, None, "a.u."
        data = np.asarray(stream.data, dtype=np.float64)
        if data.ndim == 1:
            return data, float(stream.fs_hz), stream.units or "a.u."
        return data[:, index], float(stream.fs_hz), stream.units or "a.u."

    def extract_lfp(
        self, channel_index: int | None = None
    ) -> tuple[np.ndarray | None, float | None, str]:
        stream, index = self._get_stream_and_channel(self.lfp_selection, channel_index)
        if stream is None:
            return None, None, "a.u."
        data = np.asarray(stream.data, dtype=np.float64)
        if data.ndim == 1:
            return data, float(stream.fs_hz), stream.units or "a.u."
        return data[:, index], float(stream.fs_hz), stream.units or "a.u."


@dataclass
class DepthModel:
    segment_model: SegmentModel

    @property
    def depth_mm(self) -> float | None:
        return self.segment_model.depth_mm

    @property
    def label(self) -> str:
        depth = "?" if self.depth_mm is None else f"{self.depth_mm:.3f}"
        return f"{depth} mm | {self.segment_model.segment_id}"


@dataclass
class TrajectoryModel:
    side: str | None
    trajectory: int | None
    target: str | None
    depths: list[DepthModel] = field(default_factory=list)

    @property
    def key(self) -> tuple[str | None, int | None]:
        return self.side, self.trajectory

    def display_label(self, group_by: str = "trajectory") -> str:
        base = f"{self.side or '?'}T{self.trajectory or '?'}"
        if group_by == "target":
            return f"{self.target or 'Unknown'} | {base}"
        if group_by == "side":
            return f"{self.side or '?'} | T{self.trajectory or '?'} | {self.target or 'Unknown'}"
        return f"{base} | {self.target or 'Unknown'}"


@dataclass
class CaseModel:
    session: Session
    case_dir: Path
    case_name: str
    case_metadata: dict
    trajectories: list[TrajectoryModel]
    ungrouped_segments: list[SegmentModel] = field(default_factory=list)

    def sorted_trajectories(self, group_by: str) -> list[TrajectoryModel]:
        def key(item: TrajectoryModel) -> tuple:
            if group_by == "target":
                return (item.target or "ZZZ", item.side or "Z", item.trajectory or 999)
            if group_by == "side":
                return (item.side or "Z", item.trajectory or 999, item.target or "ZZZ")
            return (item.side or "Z", item.trajectory or 999, item.target or "ZZZ")

        return sorted(self.trajectories, key=key)

    def summary_lines(self) -> list[str]:
        meta = self.case_metadata or {}
        target = meta.get("target_case_default") or "unresolved"
        notes = meta.get("notes") or []
        return [
            f"Case: {self.case_name}",
            f"Directory: {self.case_dir}",
            f"Date: {meta.get('date') or 'unknown'}",
            f"Target: {target}",
            f"Trajectories: {len(self.trajectories)}",
            f"Unmapped segments: {len(self.ungrouped_segments)}",
            f"Notes: {notes[0] if notes else 'none'}",
        ]


def build_case_model(session: Session, config: ReportConfig) -> CaseModel:
    case_dir = Path(session.meta.get("case_dir", session.session_path)).expanduser().resolve()
    case_meta = session.meta.get("case_metadata") or {}
    if not case_meta:
        parsed = resolve_case_metadata(case_dir, config)
        case_meta = {
            "xml_path": parsed.xml_path,
            "target_case_default": parsed.target_case_default,
            "target_by_side": parsed.target_by_side,
            "notes": parsed.notes,
        }
    grouped: dict[tuple[str | None, int | None], list[DepthModel]] = defaultdict(list)
    ungrouped: list[SegmentModel] = []
    resolved_case_meta = resolve_case_metadata(case_dir, config)
    for segment in session.segments:
        side = segment.meta.get("hemisphere")
        trajectory = segment.meta.get("trajectory_number")
        depth = segment.meta.get("depth_mm")
        if depth is not None:
            depth = float(depth)
        mer_selection = select_mer_stream(segment, config)
        lfp_selection = select_lfp_stream(segment, config)
        target = (
            resolve_target(str(side), int(trajectory), resolved_case_meta, config)
            if side is not None and trajectory is not None
            else None
        )
        segment_model = SegmentModel(
            segment=segment,
            side=str(side) if side is not None else None,
            trajectory=int(trajectory) if trajectory is not None else None,
            target=target,
            depth_mm=depth,
            mer_selection=mer_selection,
            lfp_selection=lfp_selection,
        )
        if side is None or trajectory is None:
            ungrouped.append(segment_model)
            continue
        grouped[(segment_model.side, segment_model.trajectory)].append(DepthModel(segment_model))

    trajectories: list[TrajectoryModel] = []
    for (side, trajectory), depths in grouped.items():
        ordered_depths = sorted(
            depths,
            key=lambda item: float("inf") if item.depth_mm is None else item.depth_mm,
        )
        target = ordered_depths[0].segment_model.target if ordered_depths else None
        trajectories.append(
            TrajectoryModel(
                side=side,
                trajectory=trajectory,
                target=target,
                depths=ordered_depths,
            )
        )
    return CaseModel(
        session=session,
        case_dir=case_dir,
        case_name=session.meta.get("case_name", case_dir.name),
        case_metadata=case_meta,
        trajectories=trajectories,
        ungrouped_segments=ungrouped,
    )
