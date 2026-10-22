<#
  Run the standard benchmark (tools/bench.py) against a running Kyojin server.

    .\tools\windows\bench.ps1                       # http://127.0.0.1:8000/v1, 3 reps
    .\tools\windows\bench.ps1 -Reps 5 -Json         # more reps, also emit the JSON block
    .\tools\windows\bench.ps1 -Base http://127.0.0.1:8001/v1

  Start the server first (tools\windows\serve.ps1), or pass -StartServer to have this script
  do it: the server is launched with the same context, waited for, benchmarked, and stopped.

  Numbers are comparable with doc/benchmarks.md only when the server runs with the packing
  documented there (ctx 32768, no --cache-bits, draft policy mix).
#>

[CmdletBinding()]
param(
    [string]$Base = "http://127.0.0.1:8000/v1",
    [int]$Reps = 3,
    [switch]$Json,
    [string]$Corpus,
    [switch]$StartServer,
    [string]$Model = "C:\P2027\port-kyo\model",
    [int]$Ctx = 32768,
    [int]$Port = 8000,
    [string]$PowerLimit
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"

function Wait-Ready([string]$Url, [int]$TimeoutS = 900) {
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        try {
            $h = Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 5
            if ($h.status -eq "ok") { return }
        } catch {}
        Start-Sleep -Seconds 5
    }
    throw "server at $Url did not become ready within $TimeoutS s"
}

$server = $null
try {
    if ($StartServer) {
        $health = "http://127.0.0.1:$Port"
        Write-Host "[bench.ps1] starting $Model (ctx $Ctx) on port $Port"
        $server = Start-Process -PassThru -NoNewWindow -FilePath "powershell" -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $PSScriptRoot "serve.ps1"),
            "-Model", $Model, "-Ctx", "$Ctx", "-Port", "$Port")
        Wait-Ready $health
        $Base = "http://127.0.0.1:$Port/v1"
    }

    $argv = @("tools\bench.py", "--base", $Base, "--reps", "$Reps")
    if ($Json) { $argv += "--json" }
    if ($Corpus) { $argv += @("--corpus", $Corpus) }
    if ($PowerLimit) { $argv += @("--power-limit", $PowerLimit) }

    & $Py @argv
    $code = $LASTEXITCODE
} finally {
    if ($server) {
        Write-Host "[bench.ps1] stopping the server"
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -like "*serve.py*--port $Port*" } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
}
exit $code
