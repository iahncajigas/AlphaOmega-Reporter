# AlphaOmega-Reporter

`alphaomega-reporter` is the command-line AlphaOmega reporting engine. It takes a case directory as input and generates PDF and HDF5 outputs without requiring the GUI application.

## Scope

- Native AlphaOmega case-directory loading
- MER and LFP trajectory-centric PDF reports
- HDF5 export
- XML target inference
- Synthetic fixtures and notebook examples
- Optional `spikeinterface` integration

This repo is the non-GUI core. The macOS desktop application lives in the separate `AlphaOmega-Reporter-App` repo and depends on this package.

## Install

Base install:

```bash
python -m pip install -e .
```

Developer install:

```bash
python -m pip install -e '.[dev]'
```

Optional extras:

```bash
python -m pip install -e '.[mpx]'
python -m pip install -e '.[spikeinterface]'
```

## Quickstart

Build a default report:

```bash
ao-report build \
  --case-dir /path/to/case_directory \
  --out report.pdf
```

Build a high-beta report with extra rendered bands:

```bash
ao-report build \
  --case-dir /path/to/case_directory \
  --out report_highbeta.pdf \
  --band highbeta \
  --plot-band alpha \
  --plot-band gamma
```

Export HDF5:

```bash
ao-report export-h5 \
  --case-dir /path/to/case_directory \
  --out session.h5
```

## Development

Validation:

```bash
make lint
make test
```

Examples and sample config are in `examples/`.
