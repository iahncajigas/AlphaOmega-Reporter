#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m PyInstaller --noconfirm --clean packaging/alphaomega_reporter_gui.spec

APP_PATH="dist/AlphaOmegaReporter.app"
ZIP_PATH="dist/AlphaOmegaReporter-macos.zip"

if [[ ! -d "$APP_PATH" ]]; then
  echo "Expected app bundle not found at $APP_PATH" >&2
  exit 1
fi

rm -f "$ZIP_PATH"
ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$ZIP_PATH"
echo "Built $APP_PATH"
echo "Packed $ZIP_PATH"
