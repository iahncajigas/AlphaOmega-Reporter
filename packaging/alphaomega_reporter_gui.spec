# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

project_root = Path.cwd()
src_root = project_root / "src"

datas = collect_data_files("alphaomega_reporter")
datas += collect_data_files("alphaomega_reporter_gui")

hiddenimports = collect_submodules("alphaomega_reporter")
hiddenimports += collect_submodules("alphaomega_reporter_gui")
hiddenimports += [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_pdf",
]

a = Analysis(
    [str(src_root / "alphaomega_reporter_gui" / "app.py")],
    pathex=[str(project_root), str(src_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "IPython",
        "ipykernel",
        "pytest",
        "notebook",
        "jupyterlab",
        "panel",
        "bokeh",
        "botocore",
        "openpyxl",
        "pandas",
        "pyarrow",
        "tables",
        "distributed",
        "dask",
        "numba",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AlphaOmegaReporter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AlphaOmegaReporter",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="AlphaOmegaReporter.app",
        bundle_identifier="org.alphaomega.reporter",
        info_plist={
            "CFBundleName": "AlphaOmegaReporter",
            "CFBundleDisplayName": "AlphaOmegaReporter",
        },
    )
