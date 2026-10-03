$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PnpmExe = "C:\Users\ROG\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd"

Push-Location $ProjectRoot
try {
    $env:SCA3_RUN_API_TESTS = "1"
    $VerificationRoot = Join-Path $ProjectRoot (".run\verification-" + [guid]::NewGuid().ToString("N"))
    $VerificationDirectory = New-Item -ItemType Directory -Path $VerificationRoot
    if (-not $VerificationDirectory.FullName.StartsWith($ProjectRoot + [IO.Path]::DirectorySeparatorChar)) { throw "Verification path is outside the project." }
    $JunitPath = Join-Path $VerificationDirectory.FullName "pytest-results.xml"
    & .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp (Join-Path $VerificationDirectory.FullName "pytest") --junitxml $JunitPath
    if ($LASTEXITCODE -ne 0) { throw "Python tests failed ($LASTEXITCODE)." }
    & .\.venv\Scripts\ruff.exe check src tests
    if ($LASTEXITCODE -ne 0) { throw "Python lint failed ($LASTEXITCODE)." }
    Push-Location "apps\web"
    try {
        & $PnpmExe build
        if ($LASTEXITCODE -ne 0) { throw "Web build failed ($LASTEXITCODE)." }
    }
    finally {
        Pop-Location
    }
    & .\.venv\Scripts\python.exe scripts\write_verification_receipt.py $JunitPath
    if ($LASTEXITCODE -ne 0) { throw "Verification receipt failed ($LASTEXITCODE)." }
}
finally {
    Pop-Location
}

Write-Output "All verification gates passed."
