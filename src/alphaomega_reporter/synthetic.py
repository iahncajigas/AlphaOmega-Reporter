from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import CaseMetadata, ContinuousStream, Segment, Session, SpikeStream


def make_synthetic_session(
    *,
    case_name: str = "synthetic_case",
    depths_mm: list[float] | None = None,
    sides: tuple[str, ...] = ("L", "R"),
    trajectory: int = 1,
    target: str = "STN",
    band_peak: str = "beta",
    random_seed: int = 0,
    missing_lfp_depths: set[float] | None = None,
    missing_spike_depths: set[float] | None = None,
) -> Session:
    rng = np.random.default_rng(random_seed)
    if depths_mm is None:
        depths_mm = [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]
    missing_lfp_depths = missing_lfp_depths or set()
    missing_spike_depths = missing_spike_depths or set()
    segments: list[Segment] = []
    for side in sides:
        for depth_mm in depths_mm:
            segment_id = f"{side}T{trajectory}D{depth_mm:+.1f}".replace(".", "p")
            streams = {}
            spikes = {}
            mer_fs = 24000.0
            lfp_fs = 1000.0
            duration_s = 2.0
            mer_values, spike_times_s, spike_unit_ids, waveforms = _make_mer_trace(
                rng,
                mer_fs,
                duration_s,
                depth_mm,
                missing=depth_mm in missing_spike_depths,
            )
            if depth_mm not in missing_spike_depths:
                streams["MER_1"] = ContinuousStream(
                    stream_id="MER",
                    fs_hz=mer_fs,
                    channel_ids=["1"],
                    channel_names=["MER 1"],
                    units="uV",
                    data=mer_values[:, None].astype(np.float32),
                )
                spikes["SEG_1"] = SpikeStream(
                    spike_id="SEG_1",
                    fs_hz=mer_fs,
                    times_s=spike_times_s,
                    unit_ids=spike_unit_ids,
                    waveforms=waveforms,
                    pre_s=0.0006,
                    post_s=0.0010,
                    source={"channel_name": "Seg 1", "channel_id": 1},
                )
            else:
                streams["MER_1"] = ContinuousStream(
                    stream_id="MER",
                    fs_hz=mer_fs,
                    channel_ids=["1"],
                    channel_names=["MER 1"],
                    units="uV",
                    data=mer_values[:, None].astype(np.float32),
                )

            if depth_mm not in missing_lfp_depths:
                lfp_values = _make_lfp_trace(rng, lfp_fs, duration_s, depth_mm, band_peak)
                streams["LFP_1"] = ContinuousStream(
                    stream_id="LFP",
                    fs_hz=lfp_fs,
                    channel_ids=["1"],
                    channel_names=["LFP 1"],
                    units="uV",
                    data=lfp_values[:, None].astype(np.float32),
                )

            segments.append(
                Segment(
                    segment_id=segment_id,
                    meta={
                        "hemisphere": side,
                        "trajectory_number": trajectory,
                        "depth_mm": depth_mm,
                        "target": target,
                        "source_file": f"{segment_id}.map",
                    },
                    streams=streams,
                    spikes=spikes,
                    events={},
                )
            )
    return Session(
        session_path=f"/synthetic/{case_name}",
        source_files=[f"/synthetic/{case_name}/{segment.segment_id}.map" for segment in segments],
        segments=segments,
        meta={
            "case_name": case_name,
            "case_dir": f"/synthetic/{case_name}",
            "case_metadata": {
                "xml_path": None,
                "target_case_default": target,
                "target_by_side": {side: target for side in sides},
                "notes": ["synthetic data"],
            },
        },
    )


def write_stub_case_dir(
    case_dir: Path | str,
    *,
    physician_description: str = "BL STN",
    include_map: bool = True,
    include_lsm: bool = True,
    include_xml: bool = True,
) -> Path:
    case_path = Path(case_dir)
    case_path.mkdir(parents=True, exist_ok=True)
    if include_map:
        for side in ("L", "R"):
            for depth in ("+1.0", "+0.0", "-1.0"):
                stem = f"{side}T1D{depth}".replace(".", "p")
                (case_path / f"{stem}.map").write_bytes(b"")
    if include_lsm:
        (case_path / "case.lsm").write_text("LT1D+1.map\nRT1D+1.map\n", encoding="utf-8")
    if include_xml:
        xml = (
            "<Root>"
            f"<PhysicianDescription>{physician_description}</PhysicianDescription>"
            f"<PatientNotes>{physician_description}</PatientNotes>"
            "</Root>"
        )
        (case_path / "case.xml").write_text(xml, encoding="utf-8")
    (case_path / ".DS_Store").write_text("", encoding="utf-8")
    (case_path / "notes.mat").write_text("", encoding="utf-8")
    (case_path / "case.xml.bak").write_text("", encoding="utf-8")
    (case_path / "conversion").mkdir(exist_ok=True)
    return case_path


def synthetic_case_metadata(case_name: str = "synthetic_case", target: str = "STN") -> CaseMetadata:
    return CaseMetadata(
        case_name=case_name,
        xml_path=None,
        target_case_default=target,
        target_by_side={"L": target, "R": target},
        notes=["synthetic"],
    )


def _make_lfp_trace(
    rng: np.random.Generator,
    fs_hz: float,
    duration_s: float,
    depth_mm: float,
    band_peak: str,
) -> np.ndarray:
    n_samples = int(round(fs_hz * duration_s))
    times = np.arange(n_samples, dtype=np.float64) / fs_hz
    peak_map = {
        "delta": 2.0,
        "theta": 6.0,
        "alpha": 10.0,
        "beta": 20.0,
        "highbeta": 25.0,
        "gamma": 45.0,
    }
    freq_hz = peak_map.get(band_peak, 20.0)
    gain = 1.0 + 2.5 * np.exp(-0.5 * (depth_mm / 1.1) ** 2)
    signal = 12.0 * rng.standard_normal(n_samples)
    signal += 6.0 * np.sin(2.0 * np.pi * 8.0 * times)
    signal += gain * 18.0 * np.sin(2.0 * np.pi * freq_hz * times)
    return signal.astype(np.float32)


def _make_mer_trace(
    rng: np.random.Generator,
    fs_hz: float,
    duration_s: float,
    depth_mm: float,
    *,
    missing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = int(round(fs_hz * duration_s))
    values = 8.0 * rng.standard_normal(n_samples)
    rate_hz = 8.0 + 35.0 * np.exp(-0.5 * (depth_mm / 0.9) ** 2)
    if missing:
        return (
            values.astype(np.float32),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, 39), dtype=np.float32),
        )

    n_spikes = int(max(4, round(rate_hz * duration_s)))
    spike_times_s = np.sort(rng.uniform(0.05, duration_s - 0.05, size=n_spikes))
    unit_ids = rng.integers(0, 2, size=n_spikes, endpoint=False, dtype=np.int32)
    waveform_len = 39
    pre = 14
    times = np.arange(waveform_len, dtype=np.float64)
    base = -np.exp(-0.5 * ((times - pre) / 2.0) ** 2)
    rebound = 0.35 * np.exp(-0.5 * ((times - (pre + 6)) / 3.0) ** 2)
    template = base + rebound
    waveforms = np.zeros((n_spikes, waveform_len), dtype=np.float32)
    for idx, spike_time_s in enumerate(spike_times_s):
        amplitude = (55.0 + 30.0 * np.exp(-0.5 * (depth_mm / 1.0) ** 2)) * (
            1.0 + 0.15 * unit_ids[idx]
        )
        waveform = amplitude * template + rng.normal(scale=2.0, size=waveform_len)
        waveforms[idx] = waveform.astype(np.float32)
        sample = int(round(spike_time_s * fs_hz))
        start = sample - pre
        stop = start + waveform_len
        if start < 0 or stop > n_samples:
            continue
        values[start:stop] += waveform
    return (
        values.astype(np.float32),
        spike_times_s.astype(np.float64),
        unit_ids.astype(np.int32),
        waveforms,
    )
