from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from alphaomega_reporter.cli import app
from alphaomega_reporter.config import ReportConfig
from alphaomega_reporter.features import ordered_bands
from alphaomega_reporter.synthetic import write_stub_case_dir


def test_ordered_bands_respects_canonical_display_order() -> None:
    ordered = ordered_bands("beta", ["gamma", "alpha", "beta"], ReportConfig())
    assert ordered == ["alpha", "beta", "gamma"]


def test_cli_accepts_gamma_band_and_plot_bands(tmp_path, monkeypatch) -> None:
    case_dir = write_stub_case_dir(tmp_path / "case")
    captured = {}

    def fake_build_report(case_dir_arg, out_arg, config):
        captured["case_dir"] = str(case_dir_arg)
        captured["out"] = str(out_arg)
        captured["primary"] = config.bands.default_primary
        captured["plot_bands"] = config.render.plot_bands
        Path(out_arg).write_text("stub", encoding="utf-8")
        return Path(out_arg)

    monkeypatch.setattr("alphaomega_reporter.cli.build_report", fake_build_report)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build",
            "--case-dir",
            str(case_dir),
            "--out",
            str(tmp_path / "report.pdf"),
            "--band",
            "gamma",
            "--plot-band",
            "alpha",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["primary"] == "gamma"
    assert captured["plot_bands"] == ["alpha"]


def test_cli_accepts_lfp_notch_options(tmp_path, monkeypatch) -> None:
    case_dir = write_stub_case_dir(tmp_path / "case")
    captured = {}

    def fake_build_report(case_dir_arg, out_arg, config):
        captured["notch_hz"] = config.lfp.noise_notch_hz
        captured["notch_width_hz"] = config.lfp.noise_notch_width_hz
        Path(out_arg).write_text("stub", encoding="utf-8")
        return Path(out_arg)

    monkeypatch.setattr("alphaomega_reporter.cli.build_report", fake_build_report)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build",
            "--case-dir",
            str(case_dir),
            "--out",
            str(tmp_path / "report.pdf"),
            "--noise-notch-hz",
            "60",
            "--noise-notch-width-hz",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["notch_hz"] == 60.0
    assert captured["notch_width_hz"] == 4.0


def test_cli_rejects_unknown_band(tmp_path) -> None:
    case_dir = write_stub_case_dir(tmp_path / "case")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "build",
            "--case-dir",
            str(case_dir),
            "--out",
            str(tmp_path / "report.pdf"),
            "--band",
            "notaband",
        ],
    )

    assert result.exit_code == 1
