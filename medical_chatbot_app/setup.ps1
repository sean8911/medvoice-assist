$ErrorActionPreference = "Stop"

$AppRoot = $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $AppRoot
$FrontendRoot = Join-Path $AppRoot "frontend"
$BackendRequirements = Join-Path $AppRoot "backend\requirements.txt"

$BundledNodeHome = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin"
if (-not (Get-Command node -ErrorAction SilentlyContinue) -and (Test-Path -LiteralPath (Join-Path $BundledNodeHome "node.exe"))) {
    $env:Path = "$BundledNodeHome;$env:Path"
}

$Python = Get-Command python -ErrorAction Stop
$Pnpm = Get-Command pnpm -ErrorAction SilentlyContinue
if ($null -eq $Pnpm) {
    $BundledPnpm = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd"
    if (Test-Path -LiteralPath $BundledPnpm) {
        $PnpmPath = $BundledPnpm
    } else {
        throw "pnpm was not found. Install Node.js and pnpm, then run setup.ps1 again."
    }
} else {
    $PnpmPath = $Pnpm.Source
}

Write-Host "Installing Python backend dependencies..." -ForegroundColor Cyan
& $Python.Source -m pip install -r $BackendRequirements

Write-Host "Installing React frontend dependencies..." -ForegroundColor Cyan
Push-Location $FrontendRoot
try {
    & $PnpmPath install
    & $PnpmPath build
} finally {
    Pop-Location
}

Write-Host "Setup complete. Run .\start_app.ps1 to open MedVoice Assist." -ForegroundColor Green
