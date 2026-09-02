param(
    [string]$EnvironmentPath = "$env:USERPROFILE\.venvs\satquery",
    [string]$HuggingFaceCache = "$env:USERPROFILE\.cache\huggingface",
    [switch]$WithLocalModel
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonCommand = Get-Command python -ErrorAction Stop

if ($EnvironmentPath -notlike "$env:USERPROFILE\*") {
    throw "EnvironmentPath must remain inside the current user profile."
}

New-Item -ItemType Directory -Force -Path $HuggingFaceCache | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $EnvironmentPath "Scripts\python.exe"))) {
    & $pythonCommand.Source -m venv $EnvironmentPath
}

$runtimePython = Join-Path $EnvironmentPath "Scripts\python.exe"
& $runtimePython -m pip install --upgrade pip
Push-Location $projectRoot
try {
    & $runtimePython -m pip install -r backend\requirements.txt -r backend\requirements-dev.txt
    if ($WithLocalModel) {
        # Optional only: Kaggle/ngrok users do not need the multi-gigabyte CUDA runtime locally.
        & $runtimePython -m pip install --upgrade `
            torch==2.6.0 torchvision==0.21.0 `
            --index-url https://download.pytorch.org/whl/cu126
        & $runtimePython -m pip install -r requirements-local-model.txt
        & $runtimePython scripts\verify-local-runtime.py
    }
    & npm install
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "SatQuery local dependencies are ready."
if (-not $WithLocalModel) {
    Write-Host "Remote model profile selected; no local CUDA/PyTorch model runtime was installed."
}
Write-Host "Model cache: $HuggingFaceCache"
Write-Host "Start everything with:"
Write-Host "  & '$runtimePython' scripts\run-local.py"
