# AlphaOmega-Reporter

`alphaomega-reporter` is a stand-alone Python toolbox for building intraoperative MER+LFP depth-to-target PDF reports from AlphaOmega case directories.

It is designed for case-directory input, native `.map/.lsm` support, XML target inference when available, and synthetic-only tests and examples. The repository does not depend on `neuroomega-io` or `nstat` at runtime.

## Features

- Native case-directory loading for AlphaOmega MAP workflows
- XML target inference from root sidecars such as `PhysicianDescription=BL STN`
- PDF reports with `matplotlib` and `PdfPages`
- Cross-platform desktop GUI built with `PySide6`
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
python -m pip install -e '.[gui]'
python -m pip install -e '.[spikeinterface]'
python -m pip install -e '.[mpx]'
python -m pip install -e '.[gui-performance]'
python -m pip install -e '.[build]'
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

Apply a configurable 60 Hz notch before the LFP PSD:

```bash
ao-report build \
  --case-dir /path/to/case_directory \
  --out report_notched.pdf \
  --noise-notch-hz 60 \
  --noise-notch-width-hz 4
```

Export a reporter-owned AO-H5-compatible file:

```bash
ao-report export-h5 \
  --case-dir /path/to/case_directory \
  --out session.h5
```

Launch the desktop GUI from source:

```bash
ao-reporter-gui
```

If `ao-reporter-gui` reports that `PySide6` is missing, install the GUI extra first:

```bash
python -m pip install -e '.[gui]'
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

Optional LFP noise suppression:

- Use `--noise-notch-hz` to set the notch center, for example `60`
- Use `--noise-notch-width-hz` to set the full width of the suppressed frequency range in Hz
- The notch is applied before PSD estimation so the heatmap, bandpower plots, and summary table stay aligned

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

## Desktop GUI

Run from source:

```bash
python -m pip install -e '.[gui]'
ao-reporter-gui
```

The GUI provides:

- Case-directory loading with the same loader stack as the CLI
- Left-side trajectory and depth selection with editable depth values
- MER preview controls for highpass, lowpass, notch, filter order, and spike detection
- LFP controls for band presets, custom bands, Welch or multitaper PSD, notch filtering, and frequency range
- A report-builder tab with per-panel inclusion checkboxes, output selection, config save/load, and PDF generation

Case loading workflow:

1. Click `Open Case...`
2. Choose a case directory
3. Select a trajectory in the left sidebar
4. Select one or more depths to drive the raster and LFP heatmap

Filter and band controls:

- MER previews use SciPy digital filters and update asynchronously
- LFP bandpower can use `delta`, `theta`, `alpha`, `beta`, `highbeta`, `gamma`, or a custom band
- Notch controls can be disabled or set to `50 Hz`, `60 Hz`, or a custom center frequency

Report selection:

- Use the `Report Builder` tab to include or exclude the cover page, raster, heatmap, bandpower, unit plots, summary table, and optional MUA/RMS profile
- `Save Config` and `Load Config` work with YAML or JSON GUI configs
- The GUI uses the same underlying report generator as the CLI

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
- Default sorter is `tridesclous2`
- Falls back to `fast` when the requested sorter path is unavailable or fails
- A clean environment may also need `numba` and `psutil` for the internal sorter path

## QC metrics

- `ISI violation ratio`: fraction of inter-spike intervals below the refractory threshold
- `Presence ratio`: fraction of equal-duration bins containing at least one spike
- `Amplitude cutoff proxy`: histogram-based missing-tail proxy from waveform p2p amplitudes

## Python API

```python
from alphaomega_reporter import ReportConfig, build_report, export_h5, load_case
from alphaomega_reporter.synthetic import make_synthetic_session

config = ReportConfig()
session = make_synthetic_session()
build_report(session, config, "synthetic_report.pdf")
```

## Packaging

PyInstaller builds must be run on the target OS:

- macOS builds must be created on macOS
- Windows builds must be created on Windows

Recommended build environment:

```bash
python -m pip install -e '.[gui,build]'
```

Build commands:

```bash
./scripts/build_macos.sh
```

```powershell
pwsh -File .\scripts\build_windows.ps1
```

The Windows script creates a fresh virtual environment by default, installs only `.[gui]` plus `pyinstaller`, checks for conflicting Qt bindings, removes them from the build venv if necessary, and then runs PyInstaller.

Packaged desktop builds intentionally omit the optional `pyqtgraph` dependency so the bundle stays deterministic and uses the built-in Matplotlib heatmap path by default.

Artifacts:

- macOS: `dist/AlphaOmegaReporter.app` and `dist/AlphaOmegaReporter-macos.zip`
- Windows: `dist/AlphaOmegaReporter/` and `dist/AlphaOmegaReporter-windows.zip`

The PyInstaller spec is in [`packaging/alphaomega_reporter_gui.spec`](/Users/iahncajigas/Library/CloudStorage/Dropbox/Research/Matlab/AlphaOmega%20Matlab%20Loader/packaging/alphaomega_reporter_gui.spec).

## Windows Build

Prerequisites:

- Python 3.10 to 3.12 installed and on `PATH`
- PowerShell 7 or Windows PowerShell
- A working C/C++ runtime on the target machine; if the packaged app fails to start on a clean host, install the Microsoft Visual C++ Redistributable for Visual Studio 2015-2022

Build from a repository checkout:

```powershell
pwsh -File .\scripts\build_windows.ps1
```

Default behavior:

1. Creates a clean `.venv-build-windows`
2. Installs `alphaomega-reporter` with `.[gui]`
3. Installs `pyinstaller`
4. Verifies `PySide6` imports and checks for `PyQt5`, `PyQt6`, and `PySide2`
5. Removes conflicting Qt bindings from the build venv if they are present
6. Builds `dist/AlphaOmegaReporter\`
7. Runs a best-effort offscreen smoke launch
8. Writes `dist/AlphaOmegaReporter-windows.zip`

If you intentionally want to use an existing interpreter, pass `-UseSystemPython`, but that mode is less deterministic and will stop if conflicting Qt bindings are installed.

Typical build time on a clean machine is under 10 minutes.

## Troubleshooting

Multiple Qt bindings error:

- PyInstaller can fail when `PySide6`, `PyQt5`, `PyQt6`, or `PySide2` are installed together
- The spec excludes non-PySide6 bindings, and the Windows build script removes conflicting bindings from its clean venv before packaging
- If you use `-UseSystemPython`, remove `PyQt5`, `PyQt6`, and `PySide2` manually or switch back to the default clean-venv flow

Qt platform plugin error:

- Rebuild from a clean environment instead of reusing a long-lived dev interpreter
- Upgrade `pyinstaller` and its bundled hooks
- Confirm the build was produced from [`packaging/alphaomega_reporter_gui.spec`](/Users/iahncajigas/Library/CloudStorage/Dropbox/Research/Matlab/AlphaOmega%20Matlab%20Loader/packaging/alphaomega_reporter_gui.spec), which bundles the PySide6 plugins explicitly through PyInstaller's Qt hooks

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

Optional SpikeInterface success-path validation:

```bash
AO_REPORTER_RUN_SPIKEINTERFACE=1 pytest -q -k spikeinterface_success
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
