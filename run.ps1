# FCL-PRM launch script (Windows PowerShell)
$ErrorActionPreference = "Stop"

$projectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { $PWD.Path }

if (-not $env:VIRTUAL_ENV) {
    $venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
    if (Test-Path $venvActivate) {
        & $venvActivate
    } else {
        Write-Error "Virtual environment not found at $venvActivate"
        exit 1
    }
}

$env:PYTHONPATH = (Join-Path $projectRoot "src")
python (Join-Path $projectRoot "scripts\run_federated.py") @args
