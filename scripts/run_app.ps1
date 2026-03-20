param(
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Error "Could not find $pythonPath. Create the virtual environment first."
}

$reloadArgs = @()
if (-not $NoReload) {
    $reloadArgs += "--reload"
}

Push-Location $projectRoot
try {
    & $pythonPath -m uvicorn --app-dir "src" "smo.web.app:app" @reloadArgs
}
finally {
    Pop-Location
}
