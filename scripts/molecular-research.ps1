param([switch]$Acquire, [switch]$Verify)
$ErrorActionPreference = "Stop"
$ResearchRoot = Split-Path -Parent $PSScriptRoot
$ResearchPython = Join-Path $ResearchRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ResearchPython)) { throw "Run setup.ps1 first." }
function Invoke-ResearchStage {
    param([string]$Module, [string[]]$StageArguments = @())
    & $ResearchPython -m $Module @StageArguments
    if ($LASTEXITCODE -ne 0) { throw "Research stage failed: $Module ($LASTEXITCODE)" }
}
Push-Location $ResearchRoot
try {
    if (Test-Path -LiteralPath "artifacts/molecular-transport-freeze.json") {
        throw "Frozen external-cohort inputs exist. This legacy pipeline would overwrite them. Use scripts/research-review.ps1 for verification; use a separate clean experiment directory for a new protocol."
    }
    # Acquisition is explicit; cached files are rehashed, not silently replaced.
    if ($Acquire) { Invoke-ResearchStage "sca3_compass.molecular_data" }
    if (-not (Test-Path -LiteralPath "artifacts/molecular-data-audit.json")) {
        throw "Acquire the registered public files first: use -Acquire."
    }
    Invoke-ResearchStage "sca3_compass.molecular_qc"
    Invoke-ResearchStage "sca3_compass.molecular_benchmark" @("--phase", "development")
    Invoke-ResearchStage "sca3_compass.molecular_benchmark" @("--phase", "validation")
    Invoke-ResearchStage "sca3_compass.molecular_shrinkage"
    Invoke-ResearchStage "sca3_compass.molecular_release"
    if ($Verify) { & (Join-Path $PSScriptRoot "verify.ps1") }
}
finally { Pop-Location }
Write-Output "Research pipeline finished. This does NOT pass originality or external-validation gates. No paper was drafted."
