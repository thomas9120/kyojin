<# Install the verified Windows gfx1151 stack, register the package, and build.
   Prerequisites: Windows 11 x64, AMD driver, C++ Build Tools, uv or Python 3.12.
   -Plan prints the steps without changing anything. Reruns reuse the venv/cache.
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 64)][int]$Jobs = 8,
    [switch]$SkipBuild,
    [switch]$Plan
)
$ErrorActionPreference = "Stop"
$SetupRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Write-Host "Kyojin Windows setup - AMD Strix Halo / gfx1151"
Write-Host "Folder: $SetupRoot"
Write-Host "1. Check Python 3.12 and Visual C++ Build Tools."
Write-Host "2. Create/reuse .venv; install torch 2.11.0 + ROCm 7.13.0 and matching SDK."
Write-Host "3. Install dependencies with tokenizers 0.22.2 and safetensors 0.8.0."
Write-Host "4. Extract HIP SDK, register Kyojin, build extension ($Jobs jobs)."
Write-Host "Allow at least 45 GB free for downloads and the unpacked SDK, plus model storage."
Write-Host "The AMD driver and Visual Studio C++ workload must already be installed."
if ($Plan) { return }
if (-not [Environment]::Is64BitOperatingSystem) { throw "Windows x64 is required" }
Set-Location -LiteralPath $SetupRoot
$SetupLogDir = Join-Path $env:LOCALAPPDATA "Kyojin\logs"
New-Item -ItemType Directory -Path $SetupLogDir -Force | Out-Null
$SetupLog = Join-Path $SetupLogDir ("setup-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".log")
Start-Transcript -Path $SetupLog | Out-Null
function Invoke-SetupStep {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE (see $SetupLog)" }
}
try {
    $SetupVswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not $SkipBuild -and -not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        $SetupVsPath = $null
        if (Test-Path -LiteralPath $SetupVswhere) {
            $SetupVsPath = & $SetupVswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        }
        if (-not $SetupVsPath) {
            throw "Install Visual Studio 2022/2026 Build Tools with Desktop development with C++ and a Windows SDK, then rerun. https://visualstudio.microsoft.com/downloads/"
        }
    }
    $SetupPython = Join-Path $SetupRoot ".venv\Scripts\python.exe"
    $SetupUv = Get-Command uv.exe -ErrorAction SilentlyContinue
    if (-not (Test-Path -LiteralPath $SetupPython)) {
        if ($SetupUv) {
            Invoke-SetupStep $SetupUv.Source @("venv", "--python", "3.12", ".venv")
        }
        elseif (Get-Command py.exe -ErrorAction SilentlyContinue) {
            Invoke-SetupStep "py.exe" @("-3.12", "-m", "venv", ".venv")
        }
        else {
            throw "Install Python 3.12 (with the py launcher) from python.org, or install uv with: winget install --id astral-sh.uv -e"
        }
    }
    Invoke-SetupStep $SetupPython @("-c", "import sys; assert sys.version_info[:2] == (3, 12), 'The Windows ROCm wheels require Python 3.12; choose a different checkout or move the existing .venv.'")
    # uv environments may omit pip. Use uv when present; otherwise bootstrap pip.
    if ($SetupUv) {
        $SetupPipExe = $SetupUv.Source
        $SetupPipPrefix = @("pip", "install", "--python", $SetupPython)
    }
    else {
        Invoke-SetupStep $SetupPython @("-m", "ensurepip", "--upgrade")
        $SetupPipExe = $SetupPython
        $SetupPipPrefix = @("-m", "pip", "install")
    }
    $SetupIndex = @("--index-url", "https://repo.amd.com/rocm/whl/gfx1151/", "--extra-index-url", "https://pypi.org/simple")
    if ($SetupUv) { $SetupIndex += @("--index-strategy", "unsafe-best-match") }
    Invoke-SetupStep $SetupPipExe ($SetupPipPrefix + $SetupIndex + @("torch==2.11.0+rocm7.13.0", "rocm[devel]==7.13.0"))
    # Pin the ROCm torch too, so requirements cannot replace it with a PyPI CUDA build.
    Invoke-SetupStep $SetupPipExe ($SetupPipPrefix + @("-r", "requirements.txt", "torch==2.11.0+rocm7.13.0", "tokenizers==0.22.2", "safetensors==0.8.0", "triton-windows", "psutil"))
    $SetupSite = Join-Path $SetupRoot ".venv\Lib\site-packages"
    $SetupHipcc = Join-Path $SetupSite "_rocm_sdk_devel\bin\hipcc.exe"
    if (-not (Test-Path -LiteralPath $SetupHipcc)) {
        $SetupArchive = @(
            (Join-Path $SetupSite "rocm_sdk_devel\_devel.tar"),
            (Join-Path $SetupSite "_rocm_sdk_devel\_devel.tar")
        ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if (-not $SetupArchive) { throw "ROCm SDK archive not found; see doc/install-windows.md section 2." }
        Invoke-SetupStep "tar.exe" @("-xf", $SetupArchive, "-C", $SetupSite)
    }
    $SetupPreviousNoCompile = $env:EXLLAMA_NOCOMPILE
    try {
        $env:EXLLAMA_NOCOMPILE = "1"
        Invoke-SetupStep $SetupPipExe ($SetupPipPrefix + @("-e", ".", "--no-build-isolation", "--no-deps"))
    }
    finally { $env:EXLLAMA_NOCOMPILE = $SetupPreviousNoCompile }
    if (-not $SkipBuild) {
        # build.ps1 uses exit; isolate it so setup can always finish its transcript.
        Invoke-SetupStep "powershell.exe" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$PSScriptRoot\build.ps1", "-Jobs", "$Jobs")
        . "$PSScriptRoot\env.ps1"
        Invoke-SetupStep $SetupPython @("-c", "import torch, exllamav3_ext; assert torch.cuda.is_available(), 'No AMD GPU available: check your driver'; print(torch.cuda.get_device_name(0)); print('Extension import OK')")
    }
    Write-Host "Setup complete. Double-click launch-gui.bat, select your EXL3 pack folder, then Start."
    if ($SkipBuild) { Write-Host "Build skipped. Run tools\windows\build.ps1 before loading a model." }
}
finally { Stop-Transcript | Out-Null }
