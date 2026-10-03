$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $ProjectRoot ".run"
$LogRoot = Join-Path $RunRoot "logs"
$PidPath = Join-Path $RunRoot "api.pid"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

if (Test-Path -LiteralPath $PidPath) {
    $ExistingPid = [int](Get-Content -LiteralPath $PidPath -Raw)
    $Existing = Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue
    if ($Existing) {
        Write-Output "SCA3 Compass is already running (PID $ExistingPid)."
        Write-Output "Open http://127.0.0.1:8000"
        exit 0
    }
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python environment not found. Run scripts\setup.ps1 first."
}

$StdoutPath = Join-Path $LogRoot "api.stdout.log"
$StderrPath = Join-Path $LogRoot "api.stderr.log"
$Process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList @("-m", "uvicorn", "sca3_compass.app:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdoutPath `
    -RedirectStandardError $StderrPath `
    -PassThru

Set-Content -LiteralPath $PidPath -Value $Process.Id -Encoding ascii
Start-Sleep -Seconds 2

try {
    $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 10
    if ($Health.status -ne "ok") {
        throw "Health endpoint returned an unexpected state."
    }
}
catch {
    Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    throw "SCA3 Compass failed to start. Review $StderrPath"
}

Write-Output "SCA3 Compass started (PID $($Process.Id))."
Write-Output "Open http://127.0.0.1:8000"

