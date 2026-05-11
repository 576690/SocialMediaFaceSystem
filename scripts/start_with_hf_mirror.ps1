$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "未找到 Python 可执行文件：$pythonExe"
}

$env:HF_ENDPOINT = "https://hf-mirror.com"

Push-Location $projectRoot
try {
    & $pythonExe -m uvicorn app:app --reload --no-access-log --log-level warning
}
finally {
    Pop-Location
}
