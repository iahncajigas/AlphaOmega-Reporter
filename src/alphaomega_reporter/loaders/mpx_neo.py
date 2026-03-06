from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..model import ContinuousStream, Segment, Session, SpikeStream
from .listfiles import read_list_file, resolve_list_entries
from .naming import FilenameParseError, parse_ao_filename, sort_key_from_meta
from .organize import group_and_concatenate_segments


class MPXReaderUnavailableError(ImportError):
    pass


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


@dataclass
class _StreamRef:
    stream_index: int
    stream_id: str
    stream_name: str


class MPXFileReader:
    def __init__(self, mpx_path: Path):
        self.mpx_path = Path(mpx_path)
        try:
            from neo.rawio import AlphaOmegaRawIO
        except Exception as exc:
            raise MPXReaderUnavailableError(
                "MPX input requires the optional 'mpx' extra. Install alphaomega-reporter[mpx]."
            ) from exc

        parent = self.mpx_path.parent
        tmp_lsx = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lsx",
            prefix="_single_mpx_",
            dir=parent,
            delete=False,
            encoding="utf-8",
        )
        tmp_path = Path(tmp_lsx.name)
        try:
            tmp_lsx.write(self.mpx_path.name + "\n")
            tmp_lsx.flush()
            tmp_lsx.close()
            self.rawio = AlphaOmegaRawIO(
                dirname=str(parent), lsx_files=[tmp_path.name], prune_channels=True
            )
            self.rawio.parse_header()
            self.header = self.rawio.header
        finally:
            tmp_path.unlink(missing_ok=True)

    def _stream_refs(self) -> list[_StreamRef]:
        refs: list[_StreamRef] = []
        for idx, row in enumerate(self.header.get("signal_streams", [])):
            refs.append(
                _StreamRef(
                    stream_index=idx,
                    stream_id=_to_text(row["id"]),
                    stream_name=_to_text(row["name"]),
                )
            )
        return refs

    def _channels_for_stream(self, stream_id: str) -> list[Any]:
        out: list[Any] = []
        for row in self.header.get("signal_channels", []):
            if _to_text(row["stream_id"]) == stream_id:
                out.append(row)
        return out

    def _resolve_stream_ref(self, requested: str) -> _StreamRef:
        req = requested.upper()
        refs = self._stream_refs()
        for ref in refs:
            if ref.stream_id.upper() == req or ref.stream_name.upper() == req:
                return ref
        for ref in refs:
            if req in ref.stream_id.upper() or req in ref.stream_name.upper():
                return ref
        raise KeyError(f"Unknown stream '{requested}'")

    def read_stream(self, stream_name: str) -> ContinuousStream:
        ref = self._resolve_stream_ref(stream_name)
        channels = self._channels_for_stream(ref.stream_id)
        raw = self.rawio.get_analogsignal_chunk(
            block_index=0,
            seg_index=0,
            i_start=0,
            i_stop=None,
            stream_index=ref.stream_index,
        )
        scaled = self.rawio.rescale_signal_raw_to_float(
            raw, dtype="float32", stream_index=ref.stream_index
        )
        data = np.asarray(scaled, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        t0_s = float(
            self.rawio.get_signal_t_start(block_index=0, seg_index=0, stream_index=ref.stream_index)
        )
        fs_hz = float(channels[0]["sampling_rate"]) if channels else float("nan")
        return ContinuousStream(
            stream_id=ref.stream_id,
            fs_hz=fs_hz,
            channel_ids=[_to_text(channel["id"]) for channel in channels],
            channel_names=[_to_text(channel["name"]) for channel in channels],
            units=_to_text(channels[0]["units"]) if channels else "a.u.",
            data=data,
            t0_s=t0_s,
            source={
                "backend": "neo_alphaomega",
                "mpx_path": str(self.mpx_path),
                "stream_name": ref.stream_name,
            },
        )

    def read_spike_channel(
        self, spike_channel_index: int, include_waveforms: bool = True
    ) -> SpikeStream:
        row = self.header["spike_channels"][spike_channel_index]
        spike_id = _to_text(row["id"]) or f"spike_{spike_channel_index}"
        timestamps = self.rawio.get_spike_timestamps(
            block_index=0, seg_index=0, spike_channel_index=spike_channel_index
        )
        times_s = np.asarray(
            self.rawio.rescale_spike_timestamp(timestamps, dtype="float64"),
            dtype=np.float64,
        )
        waveforms = None
        if include_waveforms:
            raw_waveforms = self.rawio.get_spike_raw_waveforms(
                block_index=0,
                seg_index=0,
                spike_channel_index=spike_channel_index,
                t_start=None,
                t_stop=None,
            )
            if raw_waveforms is not None:
                waveforms = np.asarray(raw_waveforms, dtype=np.float32)
                if waveforms.ndim == 3 and waveforms.shape[1] == 1:
                    waveforms = waveforms[:, 0, :]
        fs_hz = (
            float(row["wf_sampling_rate"])
            if "wf_sampling_rate" in row.dtype.names
            else float("nan")
        )
        return SpikeStream(
            spike_id=spike_id,
            fs_hz=fs_hz,
            times_s=times_s,
            waveforms=waveforms,
            source={
                "backend": "neo_alphaomega",
                "mpx_path": str(self.mpx_path),
                "spike_channel_index": spike_channel_index,
                "spike_name": _to_text(row["name"]),
            },
        )


class MPXSessionReader:
    def __init__(self, input_path: Path):
        self.input_path = Path(input_path)

    def _resolve_mpx_files(self) -> list[Path]:
        path = self.input_path
        suffix = path.suffix.lower()
        if suffix in {".mpx", ".mlx"}:
            return [path]
        if suffix == ".lsx":
            entries = read_list_file(path)
            paths = resolve_list_entries(path.parent, entries)
            return [item for item in paths if item.suffix.lower() in {".mpx", ".mlx"}]
        if path.is_dir():
            files = list(path.glob("*.mpx")) + list(path.glob("*.mlx"))
            parsed: list[tuple[tuple, Path]] = []
            for file_path in files:
                try:
                    meta = parse_ao_filename(file_path.stem)
                    parsed.append((sort_key_from_meta(meta), file_path))
                except FilenameParseError:
                    parsed.append(((9, 9, 9, "", 9), file_path))
            return [item[1] for item in sorted(parsed, key=lambda entry: entry[0])]
        raise ValueError(f"Unsupported MPX session input: {path}")

    def read(
        self,
        include_spikes: bool = True,
        include_waveforms: bool = True,
        concat_depth_files: bool = True,
    ) -> Session:
        files = self._resolve_mpx_files()
        segments: list[Segment] = []
        for file_path in files:
            reader = MPXFileReader(file_path)
            streams: dict[str, ContinuousStream] = {}
            for ref in reader._stream_refs():
                try:
                    stream = reader.read_stream(ref.stream_id)
                except Exception:
                    continue
                suffix = stream.channel_ids[0] if stream.channel_ids else ref.stream_id
                streams[f"{stream.stream_id}_{suffix}"] = stream
            spikes: dict[str, SpikeStream] = {}
            if include_spikes:
                for idx, _row in enumerate(reader.header.get("spike_channels", [])):
                    spike = reader.read_spike_channel(idx, include_waveforms=include_waveforms)
                    spikes[spike.spike_id] = spike
            try:
                filename_meta = parse_ao_filename(file_path.stem)
            except FilenameParseError:
                filename_meta = {
                    "prefix_i": False,
                    "hemisphere": None,
                    "trajectory_number": 0,
                    "depth_mm": 0.0,
                    "tag": "",
                    "file_index": 0,
                    "raw_stem": file_path.stem,
                }
            segments.append(
                Segment(
                    segment_id=file_path.stem,
                    meta={
                        **filename_meta,
                        "source_file": str(file_path),
                        "backend": "neo_alphaomega",
                    },
                    streams=streams,
                    spikes=spikes,
                    events={},
                )
            )
        if concat_depth_files:
            segments = group_and_concatenate_segments(segments)
        return Session(
            session_path=str(self.input_path),
            source_files=[str(path) for path in files],
            segments=segments,
            meta={"backend": "neo_alphaomega", "concat_depth_files": concat_depth_files},
        )
