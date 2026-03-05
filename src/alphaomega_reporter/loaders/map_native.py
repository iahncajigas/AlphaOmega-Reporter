from __future__ import annotations

import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..model import ContinuousStream, Segment, Session, SpikeStream
from .listfiles import read_list_file, resolve_list_entries
from .naming import FilenameParseError, parse_ao_filename, sort_key_from_meta
from .organize import group_and_concatenate_segments


@dataclass
class ChannelDef:
    channel_id: int
    name: str
    mode: int | None
    fs_khz: float | None
    pre_ms: float | None
    post_ms: float | None
    raw_payload_hex: str


class MAPParseError(RuntimeError):
    pass


class MAPReader:
    def __init__(self, map_path: Path):
        self.map_path = Path(map_path)
        self.channel_defs: dict[int, ChannelDef] = {}
        self.data_blocks: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        self.block_counts: Counter[str] = Counter()
        self._scanned = False

    def _extract_name(self, payload: bytes) -> str:
        text = "".join(chr(byte) if 32 <= byte < 127 else " " for byte in payload)
        known = re.search(r"(LFP\s*\d+|Spk\s*\d+|Seg\s*\d+|RAW\s*\d+|EEG\s*\d+|AUX\s*\d+)", text)
        if known:
            return known.group(1).strip()
        chunks = re.findall(r"[A-Za-z][A-Za-z0-9 _+\-]{2,}", text)
        return chunks[-1].strip() if chunks else ""

    def _parse_channel_def(self, payload: bytes) -> ChannelDef:
        if len(payload) < 2:
            raise MAPParseError(f"Channel definition payload too short in {self.map_path}")
        channel_id = (
            struct.unpack_from("<H", payload, 8)[0]
            if len(payload) >= 10
            else struct.unpack_from("<H", payload, 0)[0]
        )
        mode = struct.unpack_from("<H", payload, 4)[0] if len(payload) >= 6 else None
        fs_khz = struct.unpack_from("<f", payload, 20)[0] if len(payload) >= 24 else None
        pre_ms = struct.unpack_from("<f", payload, 24)[0] if len(payload) >= 28 else None
        post_ms = struct.unpack_from("<f", payload, 28)[0] if len(payload) >= 32 else None
        if pre_ms is not None and (not np.isfinite(pre_ms) or abs(pre_ms) > 1e6):
            pre_ms = None
        if post_ms is not None and (not np.isfinite(post_ms) or abs(post_ms) > 1e6):
            post_ms = None
        return ChannelDef(
            channel_id=channel_id,
            name=self._extract_name(payload),
            mode=mode,
            fs_khz=fs_khz,
            pre_ms=pre_ms,
            post_ms=post_ms,
            raw_payload_hex=payload.hex(),
        )

    def scan(self) -> None:
        self.channel_defs.clear()
        self.data_blocks.clear()
        self.block_counts.clear()
        with self.map_path.open("rb") as handle:
            offset = 0
            while True:
                header = handle.read(4)
                if not header:
                    break
                if len(header) < 4:
                    raise MAPParseError(
                        f"Truncated block header at offset {offset} in {self.map_path}"
                    )
                block_len, block_type_raw, _pad = struct.unpack("<HcB", header)
                block_type = block_type_raw.decode("latin1", errors="replace")
                payload_len = block_len - 4
                if payload_len < 0:
                    raise MAPParseError(
                        f"Invalid block length {block_len} at offset {offset} in {self.map_path}"
                    )
                payload = handle.read(payload_len)
                if len(payload) != payload_len:
                    if block_type == "\xff" or block_len == 0xFFFF:
                        break
                    raise MAPParseError(
                        "Unexpected EOF while reading block at offset "
                        f"{offset} in {self.map_path}: expected {payload_len} bytes, "
                        f"got {len(payload)}"
                    )
                self.block_counts[block_type] += 1
                if block_type == "2":
                    channel_def = self._parse_channel_def(payload)
                    self.channel_defs[channel_def.channel_id] = channel_def
                elif block_type == "5":
                    if len(payload) < 6:
                        raise MAPParseError(
                            f"Type '5' payload too short at offset {offset} in {self.map_path}"
                        )
                    channel_id = struct.unpack_from("<H", payload, 0)[0]
                    idx_abs = struct.unpack_from("<I", payload, len(payload) - 4)[0]
                    raw_samples = payload[2:-4]
                    if len(raw_samples) % 2:
                        raw_samples = raw_samples[:-1]
                    samples = np.frombuffer(raw_samples, dtype="<i2").copy()
                    self.data_blocks[channel_id].append((idx_abs, samples))
                offset += block_len
        self._scanned = True

    def _infer_kind(self, channel_def: ChannelDef | None) -> str:
        if channel_def is None:
            return "continuous"
        name = channel_def.name.lower()
        if name.startswith("seg"):
            return "segmented"
        return "continuous"

    def _infer_stream_id(self, channel_def: ChannelDef | None) -> str:
        if channel_def is None:
            return "RAW"
        name = channel_def.name.lower()
        if name.startswith("lfp"):
            return "LFP"
        if name.startswith("spk"):
            return "SPK"
        return "RAW"

    def materialize(self, scale_to_uv: bool = False) -> Segment:
        if not self._scanned:
            self.scan()
        try:
            filename_meta = parse_ao_filename(self.map_path.stem)
        except FilenameParseError:
            filename_meta = {
                "prefix_i": False,
                "hemisphere": None,
                "trajectory_number": 0,
                "depth_mm": 0.0,
                "tag": "",
                "file_index": 0,
                "raw_stem": self.map_path.stem,
            }

        streams: dict[str, ContinuousStream] = {}
        spikes: dict[str, SpikeStream] = {}
        for channel_id, blocks in sorted(self.data_blocks.items()):
            if not blocks:
                continue
            channel_def = self.channel_defs.get(channel_id)
            kind = self._infer_kind(channel_def)
            fs_hz = (channel_def.fs_khz * 1000.0) if (channel_def and channel_def.fs_khz) else 1.0
            name = channel_def.name if (channel_def and channel_def.name) else str(channel_id)
            sorted_blocks = sorted(blocks, key=lambda item: item[0])
            idxs = np.array([idx for idx, _samples in sorted_blocks], dtype=np.int64)
            base = int(idxs.min())
            if kind == "segmented":
                waveforms_raw = [
                    samples.astype(np.float32, copy=False) for _, samples in sorted_blocks
                ]
                max_len = max(len(wf) for wf in waveforms_raw)
                waveforms = np.full((len(waveforms_raw), max_len), np.nan, dtype=np.float32)
                for idx, waveform in enumerate(waveforms_raw):
                    waveforms[idx, : len(waveform)] = waveform
                times_s = (idxs - base).astype(np.float64) / fs_hz
                pre_s = (
                    float(channel_def.pre_ms) / 1000.0
                    if channel_def and channel_def.pre_ms is not None
                    else None
                )
                post_s = (
                    float(channel_def.post_ms) / 1000.0
                    if channel_def and channel_def.post_ms is not None
                    else None
                )
                spike_id = f"SEG_{channel_id}"
                spikes[spike_id] = SpikeStream(
                    spike_id=spike_id,
                    fs_hz=float(fs_hz),
                    times_s=times_s,
                    waveforms=waveforms,
                    pre_s=pre_s,
                    post_s=post_s,
                    source={
                        "backend": "native_map",
                        "map_path": str(self.map_path),
                        "channel_id": int(channel_id),
                        "channel_name": name,
                        "idx_abs_base": base,
                        "n_blocks": len(sorted_blocks),
                    },
                )
                continue

            end = max(idx + len(samples) for idx, samples in sorted_blocks)
            signal = np.zeros(end - base, dtype=np.float32)
            for idx_abs, samples in sorted_blocks:
                start = idx_abs - base
                stop = start + len(samples)
                signal[start:stop] = samples.astype(np.float32, copy=False)
            units = "a.u."
            if scale_to_uv:
                units = "a.u."
            stream = ContinuousStream(
                stream_id=self._infer_stream_id(channel_def),
                fs_hz=float(fs_hz),
                channel_ids=[str(channel_id)],
                channel_names=[name],
                units=units,
                data=signal[:, None],
                t0_s=0.0,
                source={
                    "backend": "native_map",
                    "map_path": str(self.map_path),
                    "channel_id": int(channel_id),
                    "channel_name": name,
                    "idx_abs_base": base,
                    "n_blocks": len(sorted_blocks),
                },
            )
            streams[f"{stream.stream_id}_{channel_id}"] = stream

        meta: dict[str, Any] = {
            **filename_meta,
            "source_file": str(self.map_path),
            "backend": "native_map",
            "block_counts": dict(self.block_counts),
            "n_channel_defs": len(self.channel_defs),
        }
        return Segment(
            segment_id=self.map_path.stem,
            meta=meta,
            streams=streams,
            spikes=spikes,
            events={},
        )


class MAPSessionReader:
    def __init__(self, input_path: Path, allow_missing: bool = False):
        self.input_path = Path(input_path)
        self.allow_missing = allow_missing

    def _resolve_map_files(self) -> list[Path]:
        path = self.input_path
        suffix = path.suffix.lower()
        if suffix == ".map":
            return [path]
        if suffix == ".lsm":
            entries = read_list_file(path)
            paths = resolve_list_entries(path.parent, entries)
            return [item for item in paths if item.suffix.lower() == ".map"]
        if path.is_dir():
            files = list(path.glob("*.map"))
            parsed: list[tuple[tuple, Path]] = []
            for file_path in files:
                try:
                    meta = parse_ao_filename(file_path.stem)
                    parsed.append((sort_key_from_meta(meta), file_path))
                except FilenameParseError:
                    parsed.append(((9, 9, 9, "", 9), file_path))
            return [item[1] for item in sorted(parsed, key=lambda entry: entry[0])]
        raise ValueError(f"Unsupported MAP session input: {path}")

    def read(self, scale_to_uv: bool = False, concat_depth_files: bool = True) -> Session:
        map_files = self._resolve_map_files()
        segments = [
            MAPReader(map_file).materialize(scale_to_uv=scale_to_uv) for map_file in map_files
        ]
        if concat_depth_files:
            segments = group_and_concatenate_segments(segments)
        return Session(
            session_path=str(self.input_path),
            source_files=[str(path) for path in map_files],
            segments=segments,
            meta={
                "backend": "native_map",
                "allow_missing": self.allow_missing,
                "concat_depth_files": concat_depth_files,
            },
        )
