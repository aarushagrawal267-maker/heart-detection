<#
.SYNOPSIS
  Prepare QRS Atlas on Windows: Python environment, PyTorch, frontend packages,
  MIT-BIH download + checksum verification, processed beat set and database.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup.ps1
  powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Cuda      # NVIDIA GPU build of PyTorch
  powershell -ExecutionPolicy Bypass -File .\setup.ps1 -SkipData  # skip dataset download

.NOTES
  Requires Python 3.11-3.13 (python.org installer with the "py" launcher recommended)
  and Node.js 18+ (LTS). Nothing is trained automatically.
#>
[CmdletBinding()]
param(
  [switch]$Cuda,
  [switch]$Cpu,
  [switch]$SkipData,
  [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$Root = $PSScriptRoot
$TorchVersion = "2.15.0"

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

function Invoke-Checked {
  param([string]$File, [string[]]$Arguments, [string]$What)
  & $File @Arguments
  if ($LASTEXITCODE -ne 0) { Fail "$What failed (exit code $LASTEXITCODE)" }
}

# ------------------------------------------------------------------ Python
Step "Locating Python 3.11-3.13"
$python = $null
$pyArgs = @()
# Probing missing interpreters writes to stderr; with "Stop" Windows PowerShell 5.1 would
# turn that into a terminating error, so probe with a relaxed preference.
$ErrorActionPreference = "Continue"
if (Get-Command py -ErrorAction SilentlyContinue) {
  foreach ($v in @("3.13", "3.12", "3.11")) {
    try { & py "-$v" -c "import sys" 2>$null | Out-Null } catch { }
    if ($LASTEXITCODE -eq 0) { $python = "py"; $pyArgs = @("-$v"); break }
  }
}
if (-not $python) {
  foreach ($cand in @("python", "python3")) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) {
      $ver = $null
      try { $ver = & $cand -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null } catch { }
      if ($ver -in @("3.11", "3.12", "3.13")) { $python = $cand; break }
    }
  }
}
$ErrorActionPreference = "Stop"
if (-not $python) { Fail "Python 3.11, 3.12 or 3.13 not found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')." }
$pyVersion = & $python @pyArgs -c "import sys; print(sys.version.split()[0])"
Ok "Using Python $pyVersion"

# ------------------------------------------------------------------ venv
Step "Creating virtual environment (.venv)"
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  Invoke-Checked $python ($pyArgs + @("-m", "venv", (Join-Path $Root ".venv"))) "Creating the virtual environment"
  Ok "Created .venv"
} else {
  Ok ".venv already exists"
}
Invoke-Checked $venvPy @("-m", "pip", "install", "--upgrade", "pip", "--quiet") "Upgrading pip"

# ------------------------------------------------------------------ PyTorch
Step "Installing PyTorch $TorchVersion"
$useCuda = $false
if ($Cuda) { $useCuda = $true }
elseif (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  Warn "NVIDIA driver detected. Re-run with -Cuda to install the GPU build (default here: CPU build)."
}
$torchOk = $false
if ($useCuda) {
  & $venvPy -m pip install "torch==$TorchVersion" --index-url https://download.pytorch.org/whl/cu126
  if ($LASTEXITCODE -eq 0) { $torchOk = $true; Ok "Installed CUDA build" }
  else { Warn "CUDA build failed to install; falling back to the CPU build." }
}
if (-not $torchOk) {
  Invoke-Checked $venvPy @("-m", "pip", "install", "torch==$TorchVersion", "--index-url", "https://download.pytorch.org/whl/cpu") "Installing PyTorch (CPU)"
  Ok "Installed CPU build"
}

# ------------------------------------------------------------------ backend deps
Step "Installing backend dependencies"
Invoke-Checked $venvPy @("-m", "pip", "install", "-r", (Join-Path $Root "backend\requirements.txt")) "Installing backend requirements"
$device = & $venvPy -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
Ok "PyTorch device: $device"

# ------------------------------------------------------------------ frontend deps
if (-not $SkipFrontend) {
  Step "Installing frontend dependencies"
  if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Fail "Node.js not found. Install the LTS version from https://nodejs.org/" }
  $nodeMajor = [int]((& node --version).TrimStart("v").Split(".")[0])
  if ($nodeMajor -lt 18) { Fail "Node.js 18 or newer is required (found $(& node --version))." }
  Push-Location (Join-Path $Root "frontend")
  try {
    if (Test-Path "package-lock.json") { & npm.cmd ci --no-audit --no-fund } else { & npm.cmd install --no-audit --no-fund }
    if ($LASTEXITCODE -ne 0) { Fail "npm install failed" }
  } finally { Pop-Location }
  Ok "Frontend packages installed (Node $(& node --version))"
}

# ------------------------------------------------------------------ database + dataset
Step "Initialising SQLite database"
Push-Location (Join-Path $Root "backend")
try {
  Invoke-Checked $venvPy @("-m", "app.cli", "init-db") "Database initialisation"
  if (-not $SkipData) {
    Step "Downloading and verifying the MIT-BIH Arrhythmia Database (~77 MB)"
    Invoke-Checked $venvPy @("-m", "app.cli", "download") "Dataset download / verification"
  } else {
    Warn "Skipped dataset download. Use the Dataset Explorer page or: .venv\Scripts\python -m app.cli download (from backend\)"
  }
} finally { Pop-Location }

Step "Setup complete"
Ok "Start the application with:  powershell -ExecutionPolicy Bypass -File .\start.ps1"
Ok "No model was trained automatically. A trained checkpoint is included in models\ if present; otherwise train one in the Model Laboratory."
