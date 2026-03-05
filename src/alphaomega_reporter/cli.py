from __future__ import annotations

import logging
from pathlib import Path

import typer

from .config import ReportConfig
from .features import ordered_bands
from .h5io import export_h5
from .io import AlphaOmegaReporterError
from .report import build_report

app = typer.Typer(add_completion=False, no_args_is_help=True)
LOGGER = logging.getLogger("alphaomega_reporter")


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(message)s", force=True)


def _load_config(config_path: Path | None) -> ReportConfig:
    return ReportConfig.from_yaml(config_path)


def _apply_cli_overrides(
    config: ReportConfig,
    *,
    band: str | None,
    plot_bands: list[str],
    group_by: str | None,
    sort_mode: str | None,
) -> ReportConfig:
    valid_bands = set(config.bands.definitions)
    valid_group_by = {"trajectory", "target", "side"}
    valid_sort_modes = {"none", "fast", "spikeinterface"}
    if band is not None:
        if band not in valid_bands:
            raise ValueError(f"Unknown band '{band}'")
        config.bands.default_primary = band
    if plot_bands:
        unknown = [item for item in plot_bands if item not in valid_bands]
        if unknown:
            raise ValueError(f"Unknown plot band(s): {', '.join(unknown)}")
        config.render.plot_bands = plot_bands
    if group_by is not None:
        if group_by not in valid_group_by:
            raise ValueError("group-by must be one of trajectory|target|side")
        config.render.group_by = group_by
    if sort_mode is not None:
        if sort_mode not in valid_sort_modes:
            raise ValueError("sort-mode must be one of none|fast|spikeinterface")
        config.sorting.mode = sort_mode
    ordered_bands(config.bands.default_primary, config.render.plot_bands, config)
    return config


@app.command("build")
def build_command(
    case_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    out: Path = typer.Option(..., file_okay=True, dir_okay=False),
    band: str = typer.Option(
        "beta", help="Primary band: delta, theta, alpha, beta, highbeta, gamma"
    ),
    plot_band: list[str] = typer.Option(None, "--plot-band", help="Additional bands to visualize"),
    group_by: str = typer.Option("trajectory", help="trajectory, target, or side"),
    sort_mode: str = typer.Option("fast", help="none, fast, or spikeinterface"),
    config_path: Path | None = typer.Option(
        None, "--config", exists=True, file_okay=True, dir_okay=False
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    _configure_logging(verbose)
    try:
        config = _load_config(config_path)
        config = _apply_cli_overrides(
            config,
            band=band,
            plot_bands=plot_band or [],
            group_by=group_by,
            sort_mode=sort_mode,
        )
        build_report(case_dir, out, config)
        typer.echo(str(out))
    except (AlphaOmegaReporterError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise typer.Exit(code=1) from exc


@app.command("export-h5")
def export_h5_command(
    case_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    out: Path = typer.Option(..., file_okay=True, dir_okay=False),
    config_path: Path | None = typer.Option(
        None, "--config", exists=True, file_okay=True, dir_okay=False
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    _configure_logging(verbose)
    try:
        config = _load_config(config_path)
        export_h5(case_dir, out, config)
        typer.echo(str(out))
    except (AlphaOmegaReporterError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
