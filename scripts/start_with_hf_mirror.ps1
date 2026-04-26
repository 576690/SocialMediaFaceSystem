$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

$env:HF_ENDPOINT = "https://hf-mirror.com"

Push-Location $projectRoot
try {
    & $pythonExe -m uvicorn app:app --reload
}
finally {
    Pop-Location
}
