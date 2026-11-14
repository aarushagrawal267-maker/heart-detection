<#
.SYNOPSIS
  Stop the QRS Atlas backend, frontend and any running training worker.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\stop.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$RunDir = Join-Path $PSScriptRoot ".run"
$stopped = 0

function Stop-Tree([int]$procId, [string]$name) {
  if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
    # /T also ends child processes (e.g. a training worker started by the backend)
    & taskkill.exe /PID $procId /T /F 2>&1 | Out-Null
    Write-Host "    stopped $name (PID $procId)" -ForegroundColor Green
    $script:stopped++
  }
}

Write-Host "==> Stopping QRS Atlas" -ForegroundColor Cyan
foreach ($svc in "backend", "frontend") {
  $pidFile = Join-Path $RunDir "$svc.pid"
  if (Test-Path $pidFile) {
    $procId = [int](Get-Content $pidFile | Select-Object -First 1)
    Stop-Tree $procId $svc
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
  }
}

# Fallback: anything still listening on our ports, but only python/node processes.
foreach ($port in 8000, 5173) {
  try {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
      $p = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
      if ($p -and $p.ProcessName -match "^(python|pythonw|node)$") { Stop-Tree $p.Id "$($p.ProcessName) on port $port" }
      elseif ($p) { Write-Host "    port $port is used by $($p.ProcessName) (PID $($p.Id)); not stopping it" -ForegroundColor Yellow }
    }
  } catch { }
}

if ($stopped -eq 0) { Write-Host "    nothing was running" } else { Write-Host "Done." -ForegroundColor Green }
