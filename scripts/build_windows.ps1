param(
    [switch]$UseSystemPython,
    [string]$Python = "python",
    [string]$VenvDir = ".venv-build-windows"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SpecPath = Join-Path $Root "packaging\alphaomega_reporter_gui.spec"
$DistDir = Join-Path $Root "dist\AlphaOmegaReporter"
$ZipPath = Join-Path $Root "dist\AlphaOmegaReporter-windows.zip"

Set-Location $Root

function Invoke-Python {
    param(
        [string[]]$Arguments
    )

    & $script:PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Arguments -join ' ')"
    }
}

function Get-QtBindingState {
    $json = & $script:PythonExe -c "import importlib.util, json; names=['PyQt5','PyQt6','PySide2']; print(json.dumps({name: bool(importlib.util.find_spec(name)) for name in names}))"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect installed Qt bindings."
    }
    return $json | ConvertFrom-Json
}

if ($UseSystemPython) {
    $PythonExe = (Get-Command $Python).Source
    Write-Host "Using system Python: $PythonExe"
}
else {
    $VenvPath = Join-Path $Root $VenvDir
    if (Test-Path $VenvPath) {
        Remove-Item $VenvPath -Recurse -Force
    }
    & $Python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment at $VenvPath"
    }
    $PythonExe = Join-Path $VenvPath "Scripts\python.exe"
    Write-Host "Created build virtual environment: $VenvPath"
}

Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Python -Arguments @("-m", "pip", "install", "-e", ".[gui]")
Invoke-Python -Arguments @("-m", "pip", "install", "pyinstaller>=6.5")

Invoke-Python -Arguments @("-c", "import PySide6; import matplotlib; print('OK', PySide6.__version__)")
$bindings = Get-QtBindingState
Write-Host ("PyQt5 {0}" -f $bindings.PyQt5)
Write-Host ("PyQt6 {0}" -f $bindings.PyQt6)
Write-Host ("PySide2 {0}" -f $bindings.PySide2)

$extraBindings = @()
foreach ($name in @("PyQt5", "PyQt6", "PySide2")) {
    if ($bindings.$name) {
        $extraBindings += $name
    }
}

if ($extraBindings.Count -gt 0) {
    if ($UseSystemPython) {
        throw "Found additional Qt bindings in the selected interpreter: $($extraBindings -join ', '). Re-run without -UseSystemPython or remove them manually."
    }
    Write-Warning "Found additional Qt bindings in the build venv: $($extraBindings -join ', '). Uninstalling them before PyInstaller runs."
    $uninstallArgs = @("-m", "pip", "uninstall", "-y")
    $uninstallArgs += $extraBindings
    Invoke-Python -Arguments $uninstallArgs
    $bindings = Get-QtBindingState
    foreach ($name in @("PyQt5", "PyQt6", "PySide2")) {
        if ($bindings.$name) {
            throw "Failed to remove conflicting Qt binding $name from the build environment."
        }
    }
}

Invoke-Python -Arguments @("-m", "PyInstaller", "--noconfirm", "--clean", $SpecPath)

if (-not (Test-Path $DistDir)) {
    throw "Expected bundle not found at $DistDir"
}

$ExePath = Join-Path $DistDir "AlphaOmegaReporter.exe"
if (-not (Test-Path $ExePath)) {
    throw "Expected GUI executable not found at $ExePath"
}

Write-Host "Running Windows smoke launch..."
$previousQtPlatform = $env:QT_QPA_PLATFORM
try {
    $env:QT_QPA_PLATFORM = "offscreen"
    $process = Start-Process -FilePath $ExePath -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 5
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        Write-Host "Smoke launch started successfully and was terminated after 5 seconds."
    }
    elseif ($process.ExitCode -eq 0) {
        Write-Host "Smoke launch exited cleanly."
    }
    else {
        Write-Warning "Smoke launch exited with code $($process.ExitCode)."
    }
}
catch {
    Write-Warning "Smoke launch failed: $_"
}
finally {
    if ($null -eq $previousQtPlatform) {
        Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
    }
    else {
        $env:QT_QPA_PLATFORM = $previousQtPlatform
    }
}

if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}

Compress-Archive -Path $DistDir -DestinationPath $ZipPath
Write-Host "Built $DistDir"
Write-Host "Packed $ZipPath"
