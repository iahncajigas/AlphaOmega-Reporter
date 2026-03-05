# Example CLI Commands

## Default beta-band report

```bash
ao-report build \
  --case-dir /path/to/case_directory \
  --out report.pdf
```

## High-beta primary summary with extra alpha and gamma plots

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

## Group by target and skip local spike sorting

```bash
ao-report build \
  --case-dir /path/to/case_directory \
  --out report_native_spikes.pdf \
  --group-by target \
  --sort-mode none
```

## Build from a YAML config

```bash
ao-report build \
  --case-dir /path/to/case_directory \
  --out report_from_config.pdf \
  --config config.yaml
```

Example `config.yaml`:

```yaml
See [`examples/config.sample.yaml`](/Users/iahncajigas/Library/CloudStorage/Dropbox/Research/Matlab/AlphaOmega%20Matlab%20Loader/examples/config.sample.yaml).
```

## Export HDF5

```bash
ao-report export-h5 \
  --case-dir /path/to/case_directory \
  --out session.h5
```
