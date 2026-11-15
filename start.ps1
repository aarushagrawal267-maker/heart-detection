<#
.SYNOPSIS
  Start the QRS Atlas backend (http://localhost:8000) and frontend (http://localhost:5173).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\start.ps1
  powershell -ExecutionPolicy Bypass -File .\start.ps1 -NoBrowser
#>
[CmdletBinding()]
param([switch]$NoBrowser, [int]$TimeoutSeconds = 90)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$Root = $PSScriptRoot
$RunDir = Join-Path $Root ".run"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Fail "Python environment missing. Run setup.ps1 first." }
$vite = Join-Path $Root "frontend\node_modules\vite\bin\vite.js"
if (-not (Test-Path $vite)) { Fail "Frontend packages missing. Run setup.ps1 first." }
$node = (Get-Command node -ErrorAction SilentlyContinue).Source
if (-not $node) { Fail "Node.js not found on PATH." }

function Test-PortInUse([int]$port) {
  try { return [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) }
  catch { return $false }
}
foreach ($p in 8000, 5173) {
  if (Test-PortInUse $p) { Fail "Port $p is already in use. Run stop.ps1 (or close the program using it) and try again." }
}

Write-Host "==> Starting backend on http://localhost:8000" -ForegroundColor Cyan
$backend = Start-Process -FilePath $venvPy `
  -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
  -WorkingDirectory (Join-Path $Root "backend") `
  -RedirectStandardOutput (Join-Path $RunDir "backend.log") `
  -RedirectStandardError (Join-Path $RunDir "backend.err.log") `
  -WindowStyle Hidden -PassThru
Set-Content -Path (Join-Path $RunDir "backend.pid") -Value $backend.Id

Write-Host "==> Starting frontend on http://localhost:5173" -ForegroundColor Cyan
$frontend = Start-Process -FilePath $node `
  -ArgumentList @("`"$vite`"", "--port", "5173", "--strictPort", "--host", "127.0.0.1") `
  -WorkingDirectory (Join-Path $Root "frontend") `
  -RedirectStandardOutput (Join-Path $RunDir "frontend.log") `
  -RedirectStandardError (Join-Path $RunDir "frontend.err.log") `
  -WindowStyle Hidden -PassThru
Set-Content -Path (Join-Path $RunDir "frontend.pid") -Value $frontend.Id

function Wait-Url([string]$url, [string]$name) {
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
      if ($r.StatusCode -eq 200) { Write-Host "    $name is up" -ForegroundColor Green; return $true }
    } catch { Start-Sleep -Milliseconds 700 }
  }
  return $false
}

$okBackend = Wait-Url "http://127.0.0.1:8000/api/health" "Backend"
$okFrontend = Wait-Url "http://127.0.0.1:5173/" "Frontend"
if (-not $okBackend) {
  Write-Host "Backend did not start. Last log lines:" -ForegroundColor Red
  Get-Content (Join-Path $RunDir "backend.err.log") -Tail 30 -ErrorAction SilentlyContinue
  & (Join-Path $Root "stop.ps1")
  exit 1
}
if (-not $okFrontend) {
  Write-Host "Frontend did not start. Last log lines:" -ForegroundColor Red
  Get-Content (Join-Path $RunDir "frontend.err.log") -Tail 30 -ErrorAction SilentlyContinue
  & (Join-Path $Root "stop.ps1")
  exit 1
}
try {
  $proxy = Invoke-WebRequest -Uri "http://127.0.0.1:5173/api/health" -UseBasicParsing -TimeoutSec 5
  Write-Host "    Frontend -> backend proxy OK" -ForegroundColor Green
} catch { Write-Host "    Warning: frontend could not reach the backend through its proxy." -ForegroundColor Yellow }

Write-Host ""
Write-Host "QRS Atlas is running (research and education use only):" -ForegroundColor Cyan
Write-Host "  Application   http://localhost:5173"
Write-Host "  API           http://localhost:8000"
Write-Host "  API docs      http://localhost:8000/docs"
Write-Host "  Logs          .run\"
Write-Host "Stop with: powershell -ExecutionPolicy Bypass -File .\stop.ps1"
if (-not $NoBrowser) { Start-Process "http://localhost:5173" }
