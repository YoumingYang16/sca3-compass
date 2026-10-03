param([switch]$SkipWebBuild)
$ErrorActionPreference = "Stop"
$ReviewRoot = Split-Path -Parent $PSScriptRoot
$ReviewPython = Join-Path $ReviewRoot ".venv\Scripts\python.exe"
function Invoke-ReviewPython {
    param([string[]]$PythonArguments)
    & $ReviewPython @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Research review command failed: $PythonArguments ($LASTEXITCODE)" }
}
Push-Location $ReviewRoot
try {
    # These recompute numerical cross-checks, never nominate new holdout genes or
    # overwrite expression/QC/fit files used in the frozen protocol.
    Invoke-ReviewPython @("scripts/check_efilter_conformance.py")
    Invoke-ReviewPython @("scripts/check_transport_reanalysis.py")
    Invoke-ReviewPython @("-m", "sca3_compass.molecular_release")
    if (-not $SkipWebBuild) { & (Join-Path $PSScriptRoot "verify.ps1") }
    Invoke-ReviewPython @("scripts/build_research_review.py")
}
finally { Pop-Location }
Write-Output "Scoped research review completed. Outputs: artifacts/research-review. No manuscript or external submission was made."
