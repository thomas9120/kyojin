<# Start the local configuration UI using the engine's existing environment. #>
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 8090,
    [string]$Config,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$GuiRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $GuiRoot
. "$PSScriptRoot\env.ps1"
$GuiPython = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $GuiPython)) { throw "Run setup-windows.bat first: no Python environment at $Venv" }
if (-not (Get-ChildItem -LiteralPath $GuiRoot -Filter "exllamav3_ext*.pyd")) {
    Write-Warning "No compiled engine extension found. The GUI can configure settings; run setup-windows.bat before loading a model."
}
$GuiArgs = @("$GuiRoot\tools\gui\server.py", "--port", "$Port")
if ($Config) { $GuiArgs += @("--config", $Config) }
if ($NoBrowser) { $GuiArgs += "--no-browser" }
& $GuiPython @GuiArgs
exit $LASTEXITCODE
