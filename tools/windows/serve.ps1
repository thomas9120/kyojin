<#
  Run an exllamav3 (kyojin) OpenAI-compatible server on Windows.

    .\tools\windows\serve.ps1                                  # Qwen pack from C:\P2027\port-kyo\model
    .\tools\windows\serve.ps1 -Model D:\packs\glm -Ctx 8192    # another pack / smaller cache
    .\tools\windows\serve.ps1 -Tool glm -Extra --no-vision     # flags are passed through

  Serves on http://127.0.0.1:<Port>/v1 (OpenAI-compatible), /health while loading.
#>

[CmdletBinding()]
param(
    [string]$Model = "C:\P2027\port-kyo\model",
    [int]$Port = 8000,
    [int]$Ctx = 32768,
    [ValidateSet("qwen", "glm", "mimo")]
    [string]$Tool = "qwen",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $Root
. "$PSScriptRoot\env.ps1"

$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "tools\$Tool\serve.py"
if (-not (Test-Path $Script)) { throw "no server at $Script" }
if (-not (Test-Path $Model)) { throw "no model directory at $Model" }

$argv = @($Script, "--model", $Model, "--port", "$Port", "-c", "$Ctx")
if ($Extra) { $argv += $Extra }

Write-Host "[serve.ps1] $Py $($argv -join ' ')"
Write-Host "[serve.ps1] listening on http://127.0.0.1:$Port  (Ctrl+C to stop)"
& $Py @argv
exit $LASTEXITCODE
