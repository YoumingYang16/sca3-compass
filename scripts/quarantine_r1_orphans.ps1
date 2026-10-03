param([Parameter(Mandatory=$true)][string]$AuditSummary)
$ErrorActionPreference = 'Stop'
$taskProject = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$taskWindow = [System.IO.Path]::GetFullPath((Join-Path $taskProject 'artifacts/robustness/R1R2-20260916'))
$taskAuditPath = (Resolve-Path -LiteralPath $AuditSummary).ProviderPath
if (-not $taskAuditPath.StartsWith($taskWindow + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Audit must belong to this research archive' }
$taskAudit = Get-Content -LiteralPath $taskAuditPath -Raw | ConvertFrom-Json
if ($taskAudit.run_id -ne 'R0073' -or $taskAudit.complete_audit -ne $true -or $taskAudit.safe_after_quarantining_listed_orphans -ne $true -or @($taskAudit.issues).Count -ne 0 -or @($taskAudit.scientific_failures).Count -ne 0) { throw 'Complete clean integrity audit required; never quarantine scientific failures automatically' }
$taskSourceRoot = (Resolve-Path -LiteralPath (Join-Path $taskProject 'artifacts/robustness/R0073-repetitions')).ProviderPath
$taskQuarantineRoot = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $taskAuditPath) 'quarantine-R0073'))
if (-not $taskQuarantineRoot.StartsWith($taskWindow + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid quarantine root' }
if (Test-Path -LiteralPath $taskQuarantineRoot) { throw 'Preserve earlier quarantine; do not overwrite/repeat moves' }
$taskMoves = [System.Collections.Generic.List[object]]::new()
foreach ($taskEntry in $taskAudit.quarantine_plan) {
    $taskSource = (Resolve-Path -LiteralPath $taskEntry.source).ProviderPath
    $taskExpectedSource = [System.IO.Path]::GetFullPath((Join-Path $taskSourceRoot $taskEntry.relative_path))
    $taskDestination = [System.IO.Path]::GetFullPath((Join-Path $taskQuarantineRoot $taskEntry.relative_path))
    if (-not [string]::Equals($taskSource,$taskExpectedSource,[StringComparison]::OrdinalIgnoreCase) -or -not $taskSource.StartsWith($taskSourceRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or -not $taskDestination.StartsWith($taskQuarantineRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Move escaped the exact validated roots' }
    if ((Get-Item -LiteralPath $taskSource).PSIsContainer -or (Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskEntry.sha256) { throw 'Source changed since audit' }
    if (Test-Path -LiteralPath $taskDestination) { throw 'Refuse destination overwrite' }
    $taskMoves.Add([pscustomobject]@{source=$taskSource; destination=$taskDestination; sha256=$taskEntry.sha256; reason=$taskEntry.reason})
}
# Every exact target is checked before any individual, non-recursive move.
[void](New-Item -ItemType Directory -Path $taskQuarantineRoot)
$taskDone = [System.Collections.Generic.List[object]]::new()
$taskMoveError = $null
try {
    foreach ($taskMove in $taskMoves) {
        $taskParent = Split-Path -Parent $taskMove.destination
        if (-not (Test-Path -LiteralPath $taskParent)) { [void](New-Item -ItemType Directory -Path $taskParent) }
        Move-Item -LiteralPath $taskMove.source -Destination $taskMove.destination
        if ((Get-FileHash -LiteralPath $taskMove.destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskMove.sha256) { throw 'Post-move checksum mismatch' }
        $taskDone.Add($taskMove)
    }
} catch { $taskMoveError = $_.Exception.Message }
[pscustomobject]@{run_id='R0073'; purpose='RECOVERABLE_ORPHAN_QUARANTINE_NOT_DATA_SELECTION'; completed=($null -eq $taskMoveError); moved=$taskDone.Count; planned=$taskMoves.Count; deleted=0; updated_utc=[DateTime]::UtcNow.ToString('o'); quarantine_root=$taskQuarantineRoot; audit_sha256=(Get-FileHash -LiteralPath $taskAuditPath -Algorithm SHA256).Hash.ToLowerInvariant(); receipts=$taskDone.ToArray(); error=$taskMoveError} | ConvertTo-Json -Depth 6
if ($null -ne $taskMoveError) { exit 2 }
