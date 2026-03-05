# AlphaOmega-Reporter

`alphaomega-reporter` is a stand-alone Python toolbox for building intraoperative MER+LFP depth-to-target PDF reports from AlphaOmega case directories.

It is designed for case-directory input, native `.map/.lsm` support, XML target inference when available, and synthetic-only tests and examples. The repository does not depend on `neuroomega-io` or `nstat` at runtime.

## Features

- Native case-directory loading for AlphaOmega MAP workflows
- XML target inference from root sidecars such as `PhysicianDescription=BL STN`
- PDF reports with `matplotlib` and `PdfPages`
- Supported bands: `delta`, `theta`, `alpha`, `beta`, `highbeta`, `gamma`
- Default primary band: `beta`
- Optional spike sorting with:
  - `none`: use native spike streams if present
  - `fast`: local threshold/PCA/clustering pipeline
  - `spikeinterface`: optional extra with automatic fallback to `fast`
- Stand-alone HDF5 export
- Synthetic fixtures for tests and demos

## Install

Base install:

```bash
python -m pip install -e .
```

Editable install with dev tools:

```bash
python -m pip install -e '.[dev]'
```

Optional extras:

```bash
python -m pip install -e '.[spikeinterface]'
python -m pip install -e '.[mpx]'
```

## Quickstart

Build a default beta-band report:

```bash
ao-report build \
  --case-dir /path/to/case_directory \
  --out report.pdf
```

Build a high-beta report and also render alpha and gamma band profiles:

```bash
ao-report build \
  --case-dir /path/to/case_directory \
  --out report_highbeta.pdf \
  --band highbeta \
  --plot-band alpha \
  --plot-band gamma \
  --group-by side \
  --sort-mode fast
```

Export a reporter-owned AO-H5-compatible file:

```bash
ao-report export-h5 \
  --case-dir /path/to/case_directory \
  --out session.h5
```

## Input model

The loader treats the case root as the canonical input surface.

- If root `.map` files are present, they are used directly.
- If root `.mpx/.mlx` files are present, they are used when the `mpx` extra is installed.
- If no direct files are present and exactly one `.lsm/.lsx` exists, the list file is resolved.
- Non-input artifacts such as `conversion/`, `.mat`, `.bak`, `.bin`, and `.DS_Store` are ignored.

Depth metadata resolution order:

1. `segment.meta["depth_mm"]`
2. Regex from `segment_id`
3. Regex from `source_file`

If no segment has usable depth metadata, the CLI exits with:

```text
no segments with depth metadata found
```

## Report panels

Each trajectory section contains two pages by default and a third page when multiple bands are requested.

Page 1:

- Spike raster vs depth
- LFP depth x frequency heatmap using Welch PSD, plotted as `10*log10(PSD)`
- Primary-band power vs depth

Page 2:

- Firing-rate scatter vs depth
- Firing-rate 2D histogram
- Waveform amplitude scatter vs depth
- Waveform amplitude 2D histogram
- Per-depth summary table
- Warning footer for missing streams or fallbacks

Optional Page 3:

- Small-multiple bandpower-vs-depth plots for all requested bands

Missing LFP or spike data render as placeholders and do not stop PDF generation.

## Frequency bands

Built-in bands:

- `delta`: `1-4 Hz`
- `theta`: `4-8 Hz`
- `alpha`: `8-13 Hz`
- `beta`: `13-30 Hz`
- `highbeta`: `20-30 Hz`
- `gamma`: `30-100 Hz`

The primary summary band defaults to `beta`. Additional bands can be rendered with repeated `--plot-band` options.

## Sort modes

`none`

- Use native spike streams if present.
- If unit labels exist, per-unit metrics are computed.
- If spike times exist but unit labels do not, spike counts are still reported and unit-level plots become placeholders.

`fast`

- Optional high-pass filtering
- MAD-based threshold detection
- Waveform extraction
- PCA features
- HDBSCAN when available, otherwise KMeans with silhouette selection

`spikeinterface`

- Optional extra
- Creates a single-channel recording extractor
- Applies basic preprocessing
- Falls back to `fast` when the requested sorter path is unavailable

## QC metrics

- `ISI violation ratio`: fraction of inter-spike intervals below the refractory threshold
- `Presence ratio`: fraction of equal-duration bins containing at least one spike
- `Amplitude cutoff proxy`: histogram-based missing-tail proxy from waveform p2p amplitudes

## Python API

```python
from alphaomega_reporter import ReportConfig, build_report, export_h5, load_case
from alphaomega_reporter.report import build_report_from_session
from alphaomega_reporter.synthetic import make_synthetic_session

config = ReportConfig()
session = make_synthetic_session()
build_report_from_session(session, "synthetic_report.pdf", config)
```

## Examples

- CLI examples: [`examples/example_cli_commands.md`](/Users/iahncajigas/Library/CloudStorage/Dropbox/Research/Matlab/AlphaOmega%20Matlab%20Loader/examples/example_cli_commands.md)
- Sample config: [`examples/config.sample.yaml`](/Users/iahncajigas/Library/CloudStorage/Dropbox/Research/Matlab/AlphaOmega%20Matlab%20Loader/examples/config.sample.yaml)
- Synthetic demo notebook: [`examples/notebook_demo.ipynb`](/Users/iahncajigas/Library/CloudStorage/Dropbox/Research/Matlab/AlphaOmega%20Matlab%20Loader/examples/notebook_demo.ipynb)

## Development

Run the checks locally:

```bash
ruff format .
ruff check .
pytest -q
```

## Release checklist

- Run `ruff format --check .`
- Run `ruff check src tests`
- Run `pytest -q`
- Build at least one synthetic PDF and inspect it visually
- Validate at least one private local case directory without committing any patient data
- If releasing with `spikeinterface`, verify the chosen sorter works in a clean environment

## Data handling

This repository must not include PHI or real patient recordings. Tests and examples use deterministic synthetic fixtures only. Real case directories are intended for local validation and are not committed.
