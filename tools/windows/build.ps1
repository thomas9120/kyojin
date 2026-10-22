<#
  Build the exllamav3 C++/HIP extension on native Windows (AMD gfx1151).

    .\tools\windows\build.ps1                 # incremental
    .\tools\windows\build.ps1 -Rebuild        # wipe build\ first
    .\tools\windows\build.ps1 -Jobs 4         # cap parallelism (default 8)
    .\tools\windows\build.ps1 -VerboseBuild   # EXLLAMA_VERBOSE (print full ninja commands)

  Produces exllamav3_ext*.pyd in the repository root, which is where ext.py looks for a
  precompiled module (PYTHONPATH is set to the root by env.ps1). Ninja/setuptools keep
  build\ incremental, so re-running after a single .cu edit recompiles just that TU.
#>

[CmdletBinding()]
param(
    [switch]$Rebuild,
    [switch]$VerboseBuild,
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
# setup.py resolves the source list and torch's hipify to abs paths relative to the CWD:
# running from anywhere else silently hipifies nothing ("At least one source is required")
Set-Location -LiteralPath $Root

. "$PSScriptRoot\env.ps1" -Build

$Py = Join-Path $Root ".venv\Scripts\python.exe"

if ($Jobs -gt 0) { $env:MAX_JOBS = "$Jobs" }
if ($VerboseBuild) { $env:EXLLAMA_VERBOSE = "1" }

if ($Rebuild) {
    $build = Join-Path $Root "build"
    if (Test-Path $build) {
        Write-Host "[build.ps1] removing $build"
        Remove-Item -LiteralPath $build -Recurse -Force
    }
    Get-ChildItem -LiteralPath $Root -Filter "exllamav3_ext*.pyd" | Remove-Item -Force
}

Write-Host "[build.ps1] python   $Py"
Write-Host "[build.ps1] hipcc    $env:HIPCXX"
Write-Host "[build.ps1] bitcode  $env:HIP_DEVICE_LIB_PATH"
Write-Host "[build.ps1] arch     $env:PYTORCH_ROCM_ARCH"
Write-Host "[build.ps1] jobs     $env:MAX_JOBS"
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
Write-Host "[build.ps1] msvc     $(if ($cl) { $cl.Source } else { 'MISSING' })"
Write-Host ""

& $Py "$Root\setup.py" build_ext --inplace
$rc = $LASTEXITCODE

Write-Host ""
$pyd = Get-ChildItem -LiteralPath $Root -Filter "exllamav3_ext*.pyd" -ErrorAction SilentlyContinue
if ($rc -ne 0) {
    Write-Host "[build.ps1] FAILED (exit $rc)" -ForegroundColor Red
    exit $rc
}
if ($pyd) {
    foreach ($p in $pyd) {
        Write-Host ("[build.ps1] {0}  {1:N1} MB" -f $p.Name, ($p.Length / 1MB)) -ForegroundColor Green
    }
    Write-Host "[build.ps1] verify: .\tools\windows\env.ps1 -Check"
}
else {
    Write-Host "[build.ps1] build reported success but no exllamav3_ext*.pyd in the repo root" -ForegroundColor Yellow
    exit 1
}
