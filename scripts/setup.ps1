$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonBootstrap = "C:\Users\ROG\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$PnpmExe = "C:\Users\ROG\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd"

Push-Location $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
        & $PythonBootstrap -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Python environment creation failed." }
    }
    & .\.venv\Scripts\python.exe -m pip install -e ".[api,dev]"
    if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
    Push-Location "apps\web"
    try {
        & $PnpmExe install
        if ($LASTEXITCODE -ne 0) { throw "Web dependency installation failed." }
        & $PnpmExe build
        if ($LASTEXITCODE -ne 0) { throw "Web build failed." }
    }
    finally {
        Pop-Location
    }
    & .\.venv\Scripts\python.exe -m sca3_compass.source_audit `
        --registry configs\data_sources.json `
        --output artifacts\source-audit.json
    if ($LASTEXITCODE -ne 0) { throw "Source audit failed; review its report before continuing." }
    & .\.venv\Scripts\python.exe -m sca3_compass.ingest_public_data
    if ($LASTEXITCODE -ne 0) { throw "Public ingestion failed." }
    & .\.venv\Scripts\python.exe -m sca3_compass.transcriptomics
    if ($LASTEXITCODE -ne 0) { throw "Transcript analysis failed." }
    & .\.venv\Scripts\python.exe -m sca3_compass.gene_expression
    if ($LASTEXITCODE -ne 0) { throw "Gene analysis failed." }
    & .\.venv\Scripts\python.exe -m sca3_compass.evidence_engine
    if ($LASTEXITCODE -ne 0) { throw "Evidence evaluation failed." }
}
finally {
    Pop-Location
}

Write-Output "Setup complete. Run scripts\start.ps1."
