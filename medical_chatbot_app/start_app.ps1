$ErrorActionPreference = "Stop"

$AppRoot = $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $AppRoot
$FrontendIndex = Join-Path $AppRoot "frontend\dist\index.html"
$LogDirectory = Join-Path $AppRoot "backend\logs"
$StandardLog = Join-Path $LogDirectory "server-output.log"
$ErrorLog = Join-Path $LogDirectory "server-error.log"
$Url = "http://127.0.0.1:8000"

if (-not (Test-Path -LiteralPath $FrontendIndex)) {
    throw "The frontend build is missing. Run .\setup.ps1 first."
}

$Python = Get-Command python -ErrorAction Stop
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null

$Server = Start-Process `
    -FilePath $Python.Source `
    -ArgumentList @("-m", "uvicorn", "medical_chatbot_app.backend.app.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $WorkspaceRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StandardLog `
    -RedirectStandardError $ErrorLog `
    -PassThru

try {
    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $Response = Invoke-WebRequest -Uri "$Url/api/health" -UseBasicParsing -TimeoutSec 2
            if ($Response.StatusCode -eq 200) {
                $Ready = $true
                break
            }
        } catch {
            if ($Server.HasExited) {
                break
            }
        }
    }

    if (-not $Ready) {
        throw "The application did not start. Check $ErrorLog"
    }

    Write-Host "MedVoice Assist is running at $Url" -ForegroundColor Green
    Write-Host "Close this PowerShell window or press Ctrl+C to stop the server."
    Start-Process $Url
    Wait-Process -Id $Server.Id
} finally {
    $RunningServer = Get-Process -Id $Server.Id -ErrorAction SilentlyContinue
    if ($null -ne $RunningServer) {
        Stop-Process -Id $Server.Id
    }
}
