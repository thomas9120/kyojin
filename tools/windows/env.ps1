<#
  Environment for exllamav3 (kyojin) on Windows, AMD Strix Halo / gfx1151.

    USAGE (from the repository root):
      . .\tools\windows\env.ps1              # runtime env: venv, PYTHONPATH, ROCm toolchain
      . .\tools\windows\env.ps1 -Build       # + MSVC environment, build knobs (for build.ps1)
      .\tools\windows\env.ps1 -Check         # report what the current shell can actually do

  Dot-source (leading dot) to keep the environment in your shell; running the script
  without it only affects that process, which is still what -Check wants.

  WHAT THIS FIXES (all verified on the target box)
  ------------------------------------------------
  T1  torch's bundled ROCm wheel ships runtime DLLs (amdhip64_7.dll, hipsparse, rocblas,
      ...) but no hipcc and no headers, and reports CUDA_HOME=None / IS_HIP_EXTENSION
      =False, so torch's cpp_extension refuses to build ("CUDA_HOME ... not set").
      We borrow the toolchain from the matching rocm-sdk-devel wheel (7.13.0, same
      version as torch's runtime) extracted at site-packages\_rocm_sdk_devel.
  T2  clang cannot find the device bitcode: it probes $ROCM_PATH\amdgcn\bitcode, but the
      wheels put it in lib\llvm\amdgcn\bitcode. HIP_DEVICE_LIB_PATH fixes it (this is
      "TRAP 4" in tools/strix_halo/env.sh, same trap, different OS).
  T3  mixer hazard: the extension links an import library (amdhip64.lib) and at load
      time binds to whichever amdhip64_7.dll the process finds. torch registers its own
      rocm_sdk directories via rocm_sdk.initialize_process(), so the build MUST use the
      import libs from the same 7.13 tree; a 7.15 devel tree links against symbols the
      7.13 runtime may not export. Hence EXL3_ROCM_SDK defaults to the wheel, not to
      C:\TheRock (which is a 7.15 build and only suitable for scratch probes).
  T4  MSVC: hipcc's msvc target needs cl.exe and the Windows SDK on PATH, and the link
      step is link.exe. Imported from vcvars64.bat through vswhere; skipped when cl.exe
      is already available.
#>

[CmdletBinding()]
param(
    [switch]$Build,
    [switch]$Check,
    [string]$RocmSdk = $env:EXL3_ROCM_SDK,
    [string]$Venv = $env:EXL3_VENV
)

$ErrorActionPreference = "Stop"

# --- locate repo root (two levels above tools\windows) ------------------------------
$Root = $env:EXL3_ROOT
if (-not $Root) {
    $Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
}
if (-not $Venv) { $Venv = Join-Path $Root ".venv" }
$VenvPy = Join-Path $Venv "Scripts\python.exe"

$script:Problems = New-Object System.Collections.Generic.List[string]

# --- venv + package path ------------------------------------------------------------
if (-not (Test-Path $VenvPy)) {
    $script:Problems.Add("no venv at $Venv (run: uv venv .venv --python 3.12)")
}
else {
    $env:PATH = "$(Join-Path $Venv 'Scripts');" + $env:PATH
    # The built exllamav3_ext*.pyd lives in the repo root, next to the package
    $env:PYTHONPATH = $Root + ($(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" }))
    $env:VIRTUAL_ENV = $Venv
}

# --- ROCm toolchain (needed by the build AND by the runtime JIT builders) ----------
if (-not $RocmSdk) { $RocmSdk = Join-Path $Venv "Lib\site-packages\_rocm_sdk_devel" }
$Hipcc = Join-Path $RocmSdk "bin\hipcc.exe"
if (-not (Test-Path $Hipcc)) { $Hipcc = Join-Path $RocmSdk "bin\hipcc.bat" }

if (-not (Test-Path $Hipcc)) {
    $script:Problems.Add("no hipcc under $RocmSdk (install: uv pip install `"rocm[devel]==7.13.0`" --index-url https://repo.amd.com/rocm/whl/gfx1151/ , then extract _devel.tar)")
    if ($Build) { throw "cannot build without a HIP toolchain" }
}
else {
    $env:ROCM_HOME = $RocmSdk
    $env:ROCM_PATH = $RocmSdk
    $env:HIP_PATH = $RocmSdk
    $env:HIPCXX = $Hipcc
    $sdkBin = Join-Path $RocmSdk "bin"
    $sdkLlmBin = Join-Path $RocmSdk "lib\llvm\bin"
    foreach ($d in @($sdkBin, $sdkLlmBin)) {
        if ((Test-Path $d) -and -not ($env:PATH -split ';' | Where-Object { $_ -eq $d })) {
            $env:PATH = "$d;" + $env:PATH
        }
    }
    # T2: clang finds the device library only through this variable on the wheel layout
    $bitcode = Join-Path $RocmSdk "lib\llvm\amdgcn\bitcode"
    if (-not (Test-Path $bitcode)) { $bitcode = Join-Path $RocmSdk "amdgcn\bitcode" }
    if (Test-Path $bitcode) {
        $env:HIP_DEVICE_LIB_PATH = $bitcode
    }
    else {
        $script:Problems.Add("no device bitcode under $RocmSdk (clang: cannot find ROCm device library)")
    }
    # Belt and braces: hipcc appends these to every compile, including the genco JIT calls
    $append = "--rocm-path=$RocmSdk"
    if ($env:HIP_DEVICE_LIB_PATH) { $append += " --rocm-device-lib-path=$($env:HIP_DEVICE_LIB_PATH)" }
    $env:HIPCC_COMPILE_FLAGS_APPEND = $append
    if (-not $env:PYTORCH_ROCM_ARCH) { $env:PYTORCH_ROCM_ARCH = "gfx1151" }
}

# --- host compiler for the .cpp TUs (build only) ------------------------------------
# torch's ninja writer compiles every .cpp TU with $env:CXX (default cl.exe). The HIP
# headers in the Windows SDK are not MSVC-parseable: amd_hip_vector_types.h declares
#     template <...> __attribute__((always_inline)) __HOST_DEVICE__
#         typename HIP_vector_base<T, n>::Native_vec_* get_native_pointer(...)
# before HIP_vector_base<T, n> is defined, and cl.exe evaluates that dependent member
# at definition time (C2988), while clang defers it. So any TU that reaches
# hip_runtime_api.h (ATen/hip/HIPContext.h, a module-loading header, ...) has to go
# through clang. clang-cl in --driver-mode=cl accepts the distutils cflags verbatim
# (/nologo /O2 /W3 /GL /MD /EHsc /I /D /FI), emits ordinary COFF objects for link.exe,
# and is part of the same ROCm tree hipcc comes from.
if ($Build) {
    $clangcl = Join-Path $RocmSdk "lib\llvm\bin\amdclang-cl.exe"
    if (Test-Path $clangcl) {
        $env:CXX = $clangcl
    }
    else {
        $script:Problems.Add("no amdclang-cl.exe under $RocmSdk; .cpp TUs will use cl.exe and fail on the HIP headers")
    }
}

# --- MSVC (build only) --------------------------------------------------------------
function Import-VcVars {
    # vcvars64.bat, not vcvarsx64.bat: the suffix is the target triple width
    param([string]$Arch = "64")
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return $true }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $false }
    $vsPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsPath) { return $false }
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars$Arch.bat"
    if (-not (Test-Path $vcvars)) { return $false }
    # `set` after vcvars gives the fully resolved environment (sets vs. toggles)
    $lines = & cmd.exe /c "`"$vcvars`" >nul 2>&1 && set"
    foreach ($line in $lines) {
        $i = $line.IndexOf("=")
        if ($i -lt 1) { continue }
        $name = $line.Substring(0, $i)
        $value = $line.Substring($i + 1)
        Set-Item -Path "env:$name" -Value $value
    }
    return [bool](Get-Command cl.exe -ErrorAction SilentlyContinue)
}

if ($Build) {
    if (Import-VcVars) { $script:MsvcOk = $true }
    else { $script:Problems.Add("MSVC environment not found (vswhere/vcvars64.bat); hipcc's msvc target needs cl.exe + Windows SDK") }
    # setuptools/distutils must not try to locate a compiler itself once vcvars ran
    $env:DISTUTILS_USE_SDK = "1"
    $env:MSSdk = "1"
    # 32 threads exist on the target box, but an unbounded ninja run on 60+ TUs starves it
    if (-not $env:MAX_JOBS) { $env:MAX_JOBS = "8" }
}

# --- -Check: prove the shell can build and run, not just that paths exist -----------
if ($Check) {
    Write-Host "[env.ps1] root  $Root"
    Write-Host "[env.ps1] venv  $Venv"
    if (Test-Path $Hipcc) {
        $ver = (& cmd.exe /c "`"$Hipcc`" --version" 2>&1 | Select-Object -First 2) -join " / "
        Write-Host "[env.ps1] hipcc $Hipcc"
        Write-Host "          $ver"
    }
    if ($env:HIP_DEVICE_LIB_PATH) { Write-Host "[env.ps1] bitcode $env:HIP_DEVICE_LIB_PATH" }
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($cl) { Write-Host "[env.ps1] msvc  $($cl.Source)" } else { Write-Host "[env.ps1] msvc  not imported (use -Build)" }
    Write-Host ""

    if (-not (Test-Path $VenvPy)) {
        Write-Host "[env.ps1] cannot check: no python at $VenvPy" -ForegroundColor Red
    }
    else {
        $probe = @'
import os, sys
try:
    import torch
except Exception as e:
    print(f"torch: IMPORT FAILED -- {e}")
    sys.exit(1)
print(f"torch {torch.__version__}  hip {torch.version.hip}  python {sys.version.split()[0]}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} / {p.gcnArchName} / warp={p.warp_size} / {p.total_memory / 2**30:.2f} GiB / {p.multi_processor_count} CUs")
    free, total = torch.cuda.mem_get_info()
    print(f"mem_get_info: {free / 2**30:.2f} free of {total / 2**30:.2f} GiB")
    try:
        a = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
        (a @ a).sum().item()
        torch.cuda.synchronize()
        print("GPU compute: OK")
    except Exception as e:
        print(f"GPU compute: FAILED -- {type(e).__name__}: {e}")
else:
    print("GPU: torch.cuda.is_available() is False")
try:
    import exllamav3_ext
    print(f"exllamav3_ext: OK ({os.path.basename(getattr(exllamav3_ext, '__file__', '?'))})")
except Exception as e:
    print(f"exllamav3_ext: NOT BUILT -- {type(e).__name__}: {e}")
try:
    import triton
    import triton.backends
    from triton.runtime import driver
    print(f"triton {triton.__version__}  backends {sorted(triton.backends.backends.keys())}")
    print(f"triton active backend: {type(driver.active).__module__}")
except Exception as e:
    print(f"triton: unavailable -- {type(e).__name__}: {e}")
'@
        $probePath = Join-Path $env:TEMP "exl3_env_check.py"
        Set-Content -Path $probePath -Value $probe -Encoding UTF8
        & $VenvPy $probePath
        Remove-Item $probePath -Force -ErrorAction SilentlyContinue
    }

    if ($script:Problems.Count -gt 0) {
        Write-Host ""
        Write-Host "[env.ps1] problems:" -ForegroundColor Yellow
        foreach ($p in $script:Problems) { Write-Host "  - $p" -ForegroundColor Yellow }
    }
}
