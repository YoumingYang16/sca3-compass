$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PidPath = Join-Path $ProjectRoot ".run\api.pid"

function Stop-ProcessTree {
    param([int]$TargetProcessId)
    $Children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $TargetProcessId" `
        -ErrorAction SilentlyContinue
    foreach ($Child in $Children) {
        Stop-ProcessTree -TargetProcessId $Child.ProcessId
    }
    Stop-Process -Id $TargetProcessId -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath $PidPath)) {
    Write-Output "SCA3 Compass is not running."
    exit 0
}

$ProcessId = [int](Get-Content -LiteralPath $PidPath -Raw)
$Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
if ($Process) {
    Stop-ProcessTree -TargetProcessId $ProcessId
}
Remove-Item -LiteralPath $PidPath -Force
Write-Output "SCA3 Compass stopped."
