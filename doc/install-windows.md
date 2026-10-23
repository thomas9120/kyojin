# Install on Windows (AMD gfx1151 / Strix Halo)

Kyojin runs natively on bare-metal Windows 11 on a Ryzen AI MAX+ 395 / Radeon 8060S
(gfx1151), with the ROCm-on-Windows wheels the AMD release index publishes for that target. No
WSL, no Docker, no Linux VM: the HIP extension is compiled by the same `hipcc` the Linux build
uses (the `rocm-sdk-devel` wheel), and Triton runs through `triton-windows`.

Measured on this path (context 32768, the shipped Qwen3.8-Flash-Next pack): prefill 1395 tok/s at
3.3K tokens, 1536 tok/s at 27K, decode 42-45 tok/s. Details and the Linux comparison:
[benchmarks-windows.md](benchmarks-windows.md). The port itself: [windows-port.md](windows-port.md).

## 1. Requirements

| | |
|---|---|
| OS | Windows 11 x64 |
| GPU | gfx1151 (`Ryzen AI MAX+ 395` / Radeon 8060S). Other gfx11.5 parts should work; the wheel index only publishes gfx1151 |
| Python | 3.12 (`uv` or the python.org installer; the ROCm wheels are cp312) |
| Build tools | Visual Studio 2022 or 2026 Build Tools with the C++ workload (`cl.exe`, `link.exe`) |
| Disk | ~10 GB for the toolchain, plus the model pack |
| RAM | see the memory section of [windows-port.md](windows-port.md): the BIOS UMA carve-out decides what the GPU can hold |

## 2. Create the environment

```powershell
git clone <this repo> kyojin
cd kyojin
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install --index-strategy unsafe-best-match `
    --extra-index-url https://repo.amd.com/rocm/whl/gfx1151/ `
    torch==2.11.0+rocm7.13.0
uv pip install "rocm[devel]==7.13.0"        # the matching HIP toolchain
uv pip install -e . --no-build-isolation
uv pip install "tokenizers==0.22.2" "safetensors==0.8.0"
```

Two things about those pins:

- **torch and `rocm[devel]` must be the same ROCm version** (7.13.0 here). torch loads
  `amdhip64_7.dll` from the wheel; a `hipcc` from a different major version still compiles, but the
  device bitcode and the runtime headers drift apart in ways that only show up at run time.
- **`tokenizers` and `safetensors` must be pinned.** Unpinned resolution currently picks release
  candidates (`tokenizers` 1.0.0rc, `safetensors` 0.9.0rc) and `tokenizers` 1.x no longer exports
  `models`, so `import exllamav3` fails.

The `rocm[devel]` wheel unpacks a 17 GB archive lazily; if `import rocm_sdk` complains that the
devel tree is missing, unpack it once:

```powershell
tar -xf .venv\Lib\site-packages\rocm_sdk_devel\_devel.tar -C .venv\Lib\site-packages
```

## 3. Build

```powershell
. .\tools\windows\env.ps1 -Build     # dot-source: it sets variables in your shell
.\tools\windows\build.ps1            # ~10-15 min the first time, ~10 s when nothing changed
```

`env.ps1` sets `ROCM_HOME`/`HIP_PATH`, `HIP_DEVICE_LIB_PATH`, `PYTORCH_ROCM_ARCH=gfx1151`,
`MAX_JOBS`, imports the MSVC environment from `vcvars64.bat`, and sets `CXX=amdclang-cl.exe`.
That last one matters: `cl.exe` cannot parse the ROCm SDK's HIP headers, so the `.cpp`
translation units are compiled by the clang driver in MSVC-compatibility mode instead.

`. .\tools\windows\env.ps1 -Check` prints what the shell can actually see (torch, GPU, extension,
triton, compiler) — run it whenever a build or a JIT kernel misbehaves.

`build.ps1` leaves the extension as `exllamav3_ext.cp312-win_amd64.pyd` in the repository root;
`-Rebuild` forces a clean build, `-Jobs N` changes the compile parallelism.

### Before committing

The tree is LF. If an editor or `git config core.autocrlf=true` rewrote a file to CRLF, the shell
scripts and the HIP sources break, and the diff hides the real change:

```powershell
python tools\windows\check_eol.py          # exit 1 and list offenders
python tools\windows\check_eol.py --fix
```

Two GitHub workflows cover the same ground for the port: `windows-cpu-tests.yml` runs the GPU-free
tests on Windows and Linux on every push, and `windows-rocm-build.yml` (dispatch-only) compiles the
extension on a clean `windows-2022` runner - a compile check, since a hosted runner has no AMD GPU.

## 4. Run

```powershell
.\tools\windows\serve.ps1 -Model C:\packs\qwen -Ctx 32768
```

It serves the OpenAI API on `http://127.0.0.1:8000/v1` and `/health` while loading. Use
`-Tool glm|mimo` for those servers, `-Port` and `-Ctx` for a different port or cache size, and
anything after `-Extra` is passed straight to the Python server:

```powershell
.\tools\windows\serve.ps1 -Extra --no-vision
```

Two notes:

- **Use `127.0.0.1`, not `localhost`.** On Windows `localhost` resolves to `::1` first; the server
  listens on IPv4 only, and the connect then stalls about 2 s per request (measured: the same
  server benchmarks 765 tok/s through `localhost` and 1395 through `127.0.0.1`).
- The first start compiles and tunes kernels. Expect ~8 minutes once (the dense GEMM tuner alone is
  867 rocBLAS calls, about 7 minutes on this GPU); the results are cached in
  `%LOCALAPPDATA%\.cache\exllamav3`, and later starts take about 50 s.

## 5. Benchmark

```powershell
.\tools\windows\bench.ps1                       # against a running server
.\tools\windows\bench.ps1 -StartServer          # start, wait for /health, benchmark, stop
python tools\windows\bench_context.py           # prefill/decode scaling from 2K to 32K
```

## 6. What is different from Linux

The engine code is the same; only the platform plumbing differs.

| | Linux | Windows |
|---|---|---|
| Toolchain | `/opt/rocm`, `hipcc` from PATH | `rocm-sdk-devel` wheel tree, `hipcc.exe` resolved by `exllamav3/util/hip_compiler.py` |
| Host compiler | gcc/clang | `amdclang-cl.exe` (see above) |
| Device bitcode | `$ROCM_PATH/lib/llvm/amdgcn/bitcode` | `HIP_DEVICE_LIB_PATH` into the wheel tree (`env.ps1` sets it) |
| JIT cache | `~/.cache/exllamav3` | `%LOCALAPPDATA%\.cache\exllamav3` |
| N-gram table | RAM if it fits, else disk | same rule; on a 32 GiB host it always streams from disk |
| CPU expert offload | worker + memfd arena | worker + anonymous mappings (the pinned memfd arena is off; `EXL3_MOE_PINNED_ARENA` stays 0) |
| Signals | SIGINT/SIGTERM handlers | `os.register_at_fork` and signal handlers are guarded; close the console or `Stop-Process` |

Everything else — the HIP kernels, the Triton attention, speculative decoding, the n-gram draft,
the prompt cache, `--cache-bits 8` — is the same code path as on Linux.

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ImportError: DLL load failed while importing exllamav3_ext` | torch must be imported first (it calls `os.add_dll_directory` for the ROCm DLLs). Import `exllamav3`/`torch` before the extension, and use the venv's python |
| `clang: cannot find ROCm device library` | `HIP_DEVICE_LIB_PATH` unset: dot-source `tools\windows\env.ps1` |
| `error C2988`/`C2059` in `amd_hip_vector_types.h` | the build picked `cl.exe`; dot-source `env.ps1 -Build` so `CXX=amdclang-cl.exe` |
| `LNK2001: hipblas*` / `__truncsfhf2` | the extension was built without the wheel's import libraries; build through `tools\windows\build.ps1` |
| `TypeError` in `object_file_path` | torch's hipify mangles Windows paths; `exllamav3/util/win_hip.py` patches it at import time — make sure you are on this branch |
| Every request takes ~2 s longer than the server reports | `localhost` instead of `127.0.0.1` |
| First start takes ~8 minutes | dense GEMM tuning, one time; the cache lives in `%LOCALAPPDATA%\.cache\exllamav3` |
| `AttributeError: module 'os' has no attribute 'register_at_fork'` | an older checkout; the guard is in `tools/startup_health.py` |

Extension build logs: run `build.ps1 -VerboseBuild`. Runtime JIT kernels are compiled at first use
and cached; their compile lines appear in the server log with the resolved `hipcc` path.
