Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

python -m PyInstaller --noconfirm --clean packaging/alphaomega_reporter_gui.spec

$DistDir = Join-Path $Root "dist\AlphaOmegaReporter"
$ZipPath = Join-Path $Root "dist\AlphaOmegaReporter-windows.zip"

if (-not (Test-Path $DistDir)) {
    throw "Expected bundle not found at $DistDir"
}

if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}

Compress-Archive -Path $DistDir -DestinationPath $ZipPath
Write-Host "Built $DistDir"
Write-Host "Packed $ZipPath"
