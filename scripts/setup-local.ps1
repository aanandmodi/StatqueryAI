param(
    [string]$EnvironmentPath = "$env:USERPROFILE\.venvs\satquery-cloud",
    [string]$HuggingFaceCache = "$env:USERPROFILE\.cache\huggingface",
    [string]$PythonExecutable = "",
    [switch]$WithLocalModel
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed (exit $LASTEXITCODE). Setup did not complete."
    }
}

if (-not $PythonExecutable) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $PythonExecutable = $pythonCommand.Source }
}
if (-not $PythonExecutable) {
    throw "Install Python 3.12 (Add python.exe to PATH), reopen PowerShell, or pass -PythonExecutable 'C:\path\to\python.exe'."
}
Invoke-Checked $PythonExecutable @('-c', 'import sys; assert (3,11) <= sys.version_info[:2] < (3,14), "Use Python 3.11-3.13, preferably 3.12"')
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "Install Node.js 22 LTS or newer and reopen PowerShell."
}
Invoke-Checked 'node' @('-e', 'const [major, minor] = process.versions.node.split(".").map(Number); if (major < 22 || (major === 22 && minor < 13)) { throw new Error("Node.js 22.13 or newer is required"); }')
Invoke-Checked 'npm.cmd' @('--version')

if ($EnvironmentPath -notlike "$env:USERPROFILE\*") {
    throw "EnvironmentPath must remain inside the current user profile."
}

if (-not (Test-Path -LiteralPath (Join-Path $EnvironmentPath "Scripts\python.exe"))) {
    Invoke-Checked $PythonExecutable @('-m', 'venv', $EnvironmentPath)
}

$runtimePython = Join-Path $EnvironmentPath "Scripts\python.exe"
try {
    Invoke-Checked $runtimePython @('-c', 'import sys; print("Using Python", sys.version.split()[0])')
}
catch {
    throw "Existing environment is broken. It was preserved. Rerun with -EnvironmentPath '$env:USERPROFILE\.venvs\satquery-cloud-new' after installing Python 3.12."
}
Invoke-Checked $runtimePython @('-m', 'pip', 'install', '--upgrade', 'pip')
Push-Location $projectRoot
try {
    Invoke-Checked $runtimePython @('-m', 'pip', 'install', '-r', 'backend\requirements-dev.txt')
    if ($WithLocalModel) {
        # Optional only: Kaggle/ngrok users do not need the multi-gigabyte CUDA runtime locally.
        New-Item -ItemType Directory -Force -Path $HuggingFaceCache | Out-Null
        Invoke-Checked $runtimePython @('-m', 'pip', 'install', 'torch==2.6.0', 'torchvision==0.21.0', '--index-url', 'https://download.pytorch.org/whl/cu126')
        Invoke-Checked $runtimePython @('-m', 'pip', 'install', '-r', 'requirements-local-model.txt')
        Invoke-Checked $runtimePython @('scripts\verify-local-runtime.py')
    }
    Invoke-Checked 'npm.cmd' @('ci')
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "SatQuery local dependencies are ready."
if (-not $WithLocalModel) {
    Write-Host "Remote model profile selected; no local CUDA/PyTorch model runtime was installed."
}
Write-Host "Start everything with:"
Write-Host "  & '$runtimePython' scripts\run-local.py --mode remote"
