# Kyojin on Windows: port plan (AMD gfx1151 / Strix Halo)

Status: gates A, B, D measured on the target box; Phase 1 not started. Branch: `windows-port`.

## 1. Scope

Target: bare-metal Windows 11, Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151), 128 GB, AMD
ROCm-on-Windows wheels, Python 3.12, the shipped EXL3 packs (Qwen3.8-Flash-Next first).

In scope

- the HIP code path (this fork's only path) built and running natively on Windows,
- Qwen serve bring-up (`tools/qwen/serve.py`), `tools/bench.py`, the ngram SSD-streaming path,
- the memory-critical paths: model load, KV cache, CPU/SSD offload tiers,
- performance measurement against the published Linux figures.

Out of scope for v1

- DeepSeek-V4.1-Flash SSD expert tier (`exllamav3/dsv41/**`), Nix, Docker, multi-GPU tensor
  parallelism (`model/model_tp_fn.py`), conversion tooling (`conversion/convert_model.py`),
- a Vulkan backend (see section 5).

Hard rule: **Linux must not change behaviour.** Every edit is gated by `os.name == "nt"` (and by
`torch.version.hip` only where the HIP path is what differs). The existing CUDA-on-Windows path
stays as it is.

## 2. Status: what is already proven on this box

| Thing | Result |
|---|---|
| GPU | `AMD Radeon(TM) 8060S Graphics`, `gfx1151`, 20 CU, warp 32, integrated, **107.87 GiB** visible |
| torch | `2.11.0+rocm7.13.0` (win_amd64, HIP 7.13.99004), Python 3.12.10 venv at `~/.unsloth/studio/unsloth_studio` |
| HIP toolchain | TheRock Windows build at `C:\TheRock\build`: HIP 7.15.26333, AMD clang 23, target `x86_64-pc-windows-msvc`; `hipcc.exe`, `hipify-clang.exe`, all headers (`hipsparse`, `thrust`, `hipcub`, `hipblaslt`, `rocblas`, `miopen`), bitcode at `lib\llvm\amdgcn\bitcode` |
| MSVC | BuildTools 2022 `14.44.35207` (`Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`) |
| Triton | `triton-windows 3.7.1.post27`, backends `['amd', 'nvidia']`, **active backend = AMD `HIPDriver`**; elementwise and `tl.dot` kernels compile and run (matmul max err 0.0625 fp16) |
| torch HIP extension | one-TU extension builds and runs, `BUILD+LOAD OK in 79 s`, kernel output correct — needs the two fixes in section 8 |
| Compute | fp16 33.7 TFLOP/s, bf16 30.3 TFLOP/s at 4096³ (Linux reference ~31) — no Windows penalty |
| Memory | see section 4 |

So: triton-windows already carries the AMD backend on Windows, which removes the biggest
performance risk this plan originally had. The port is a build + plumbing job.

## 3. What we start from in the repo

This is not a CUDA→HIP port; the HIP path already exists and is what the Linux machines run.

1. The CUDA side of this fork already builds and runs on Windows (inherited from ExLlamaV3):
   `setup.py` Windows flags, `cuda_drv.cpp` `nvcuda.dll` branch, `ngram_gather_win.cpp`,
   Windows branches in `loader/safetensors.py`, `model/moe_cpu_host.py`,
   `modules/ngram_embedding.py`, `cpu/moe_mul1.cpp` (`WaitOnAddress`,
   `SetThreadGroupAffinity`), and `cuda_host_get_device_pointer` already documents the WDDM
   host-alias rule.
2. The ROCm build is small and curated: `exllamav3/exllamav3_ext/build_config.py` compiles **71**
   TUs for ROCm (vs ~200 for CUDA) — EXL3 int8 GEMV, `hgemm`, `reconstruct`, MoE fused,
   `exl3_dec`, the CPU MoE handoff. CUDA-only `attention.cu`, `exl3_gemm.cu`, `exl3_moe*.cu`,
   `norm.cu`, `libtorch/*.cpp` stay excluded.
3. Only two POSIX includes exist in that set (`ngram.cu`: `<unistd.h>`; `hgemm.cu`: `<sys/stat.h>`).

Linux-only code that still needs porting (runtime path):

| Area | File | Problem on Windows |
|---|---|---|
| HIP runtime discovery | `util/hip_lib.py` | hardcodes `libamdhip64.so*`, reads `/proc/self/maps` |
| HIP compiler discovery | `util/hip_compiler.py` | candidates are `bin/hipcc`, `/opt/rocm/bin/hipcc`; needs `.exe`/`.bat` |
| JIT kernels (7 builders) | `modules/gr_fn/gr_mix_hip.py`, `modules/ple_fn/ple_hip.py`, `modules/attention_fn/qsa_prefill_hip.py`, `modules/dsv41_fn/*.py`, `vendor/fla/hip/*.py` | `--gcc-install-dir=/usr/lib/gcc/...`, `~/.cache` path, `.hsaco` cache naming |
| Host memory tuning | `util/memory.py` (`malloc_trim`, `set_host_mmap_threshold`, `swap_in_process`) | glibc-only, `/proc/self/pagemap` |
| Shared memory | `util/shm.py` | `/dev/shm` + `os.statvfs` |
| MoE CPU tier | `model/moe_cpu_host.py` | `multiprocessing.shared_memory` capacity check, memfd/`MFD_HUGE_*` arena |
| SSD expert tier (v2) | `modules/expert_store.py` | `O_DIRECT` open, `preadv`, `/proc/meminfo` |
| `/proc/meminfo` readers | `modules/expert_store.py`, `modules/ngram_embedding.py`, `dsv41/prefix_cache.py` | use `util.memory.host_memory_available()` (already psutil-fallback) |
| Server | `tools/{qwen,glm,mimo}/serve.py` | signal handling |
| Repo layout | 14 git **symlinks** (`tools/*/serve_ping.py`, `serve_reqlog.py`, `serve_metrics.py`, `image_cache.py`, `env.sh`, `bench_mtp.py`, `load_qwen.py`) | clone has `core.symlinks=false` → they are text files, imports break |
| Scripts | `build.sh`, `tools/strix_halo/*.sh`, `tools/bench.sh` | bash only |

## 4. Memory model on this box (the real constraint)

The machine is split at boot:

| Pool | Size | Notes |
|---|---|---|
| BIOS UMA carve-out (GPU dedicated) | ~96 GiB | Windows only sees **31.6 GiB** of system RAM |
| GPU shared/GTT | ~12 GiB | borrowed from the Windows side |
| **GPU visible to HIP** | **107.87 GiB** | `mem_get_info` total, stable regardless of host load |
| GPU actually usable | **~100 GiB** | 100 GiB of 4 GiB blocks ok, 104 fails; a single block caps at 88 GiB |
| Host RAM usable for offload / ngram-in-RAM | ~19 GiB free of 31.6 | `EXL3_HOST_MEM_RESERVE_MB` default 2 GiB |

Consequence: the model budget is fixed by the BIOS setting, not by desktop bloat — closing apps
does not add GPU memory. Raising the carve-out starves Windows; lowering it starves the GPU.
Single allocations over 88 GiB fail even with 107.7 GiB free (`OCL_SET_SVM_SIZE=262144` does not
change it), but the engine allocates per tensor, so the 88 GiB single-block limit does not bind.

**Qwen3.8-Flash-Next pack (measured on disk here):**

| Part | Size | Resident on GPU? |
|---|---|---|
| `model-0000X-of-00009.safetensors` | 63.34 GiB | yes |
| n-gram embedding table | 24.44 GiB | **no** — streamed from SSD per forward |
| `draft2s.st` + `hc_gs_sidecar` | 1.04 GiB | draft head (GPU) |
| **total pack** | **88.82 GiB** (95.4 GB) | **~64.4 GiB resident** |

The n-gram table goes to disk automatically on this box: `NGramEmbedding` holds it in RAM only
when `available >= table + 48 GiB` (it is 24.44 + 48 vs ~19 GiB free), so `EXL3_NGRAM_STREAM`
resolves to disk without any flag. That leaves ~35 GiB of the ~100 GiB budget for KV cache,
recurrent state and activations — comfortable even at long context.

| Pack | Size | Fit on ~100 GiB |
|---|---|---|
| Qwen3.8-Flash-Next | 88.82 GiB (64.4 resident) | **yes**, with room for long context |
| GLM-5.3-Flash | ~92.9 GiB, no ngram table | tight; likely needs CPU expert offload or less context |
| MiMo-V2.6-Flash | ~98.5 GiB | no; needs CPU expert offload |
| DeepSeek-V4.1-Flash | 172 GiB | out of scope (SSD tier) |

## 5. Why not Vulkan

Vulkan is not a port, it is a rewrite, and it does not buy the memory this box is missing.

- **Kernel surface**: the ROCm build alone is 60 GPU TUs / ~22k lines, 23 of them using AMD inline
  asm or intrinsics (`v_dot*`, `__builtin_amdgcn_*`, MFMA/WMMA), plus ~9k lines of Triton in
  `modules/attention_fn/`. All of it is HIP-native: `.hsaco` code objects loaded through
  `hipModuleLoad`, GCN-specific GEMV/GEMM kernels that are the whole point of this fork, HIP graph
  capture, cooperative kernels. A Vulkan backend means re-expressing that in GLSL/SPIR-V, plus a
  Vulkan tensor backend in place of torch (torch has no Vulkan backend).
- **EXL3 has no Vulkan consumer**: llama.cpp does not read EXL3 trellis packs, so "use Vulkan"
  really means "abandon this engine and re-quantise to GGUF". If that is ever the answer, it is a
  product decision, not a port.
- **Memory**: the ~108 GiB ceiling is the BIOS UMA split, not a HIP limitation. Vulkan/WDDM can
  additionally allocate host-visible buffers out of the ~31.6 GiB Windows side (that is where
  llama.cpp's 115-125 GB total comes from: ~96 GiB VRAM + host), so Vulkan might reach ~20 GiB more
  in total. Against that: ~31k lines of GPU code, and the shared-memory path is slower than the
  carve-out for weights. Cheaper ways to get those 20 GiB, in order: CPU expert offload (section 6),
  n-gram table on SSD (already automatic), smaller context, then a driver/BIOS experiment.

Decision: no Vulkan backend. Keep it out of the plan; revisit only if the goal changes from
"port Kyojin" to "have any fast EXL3-ish runtime on Windows".

## 6. CPU and SSD offload: what the engine actually has

The EXL3 format itself is neutral about placement; the engine ships three placing mechanisms.

1. **CPU MoE expert offload** — the direct answer to "does it support CPU offload": **yes**, for
   routed experts.
   - `-mcl/--moe_cpu_offload N`: the routed experts of the first N block-sparse MoE layers live in
     system RAM and are computed on the CPU.
   - `-mcs/--moe_cpu_split N`: the tail N experts of *every* eligible layer live on the CPU, so the
     CPU GEMMs overlap that layer's GPU expert compute.
   - Requirements: layer-split mode (the default), `mul1` codebook, K ≤ 8, uniform per-expert
     biases. The Qwen pack is `"codebook": "mul1"` ✓.
   - Implementation: a spawned worker process per model component owns the expert weights and a job
     ring in pinned shared memory (`model/moe_cpu_host.py`); the CPU kernels are AVX-512 VNNI/AVX2
     (`exllamav3_ext/cpu/moe_mul1.cpp`); the GPU publish/wait flag kernels have a ROCm port
     (`cpu/moe_handoff_amdgpu.cu`) and the bindings are registered for ROCm
     (`bindings.cpp` under `USE_ROCM`). `MoeCpuHost` only raises when those bindings are missing.
   - Known ROCm limitations (already coded around): no `cuda_stream_wait_value` memops path, static
     hot/cold placement (`EXL3_MOE_CPU_SWAP` is forced off on HIP), no fused split issue/collect.
     Slower than the CUDA path, functional.
   - Windows needs: the shared-memory segment (works; `SharedMemory` is named memory on Windows),
     the capacity check (`util/shm.py` returns `None`, already), host pinning
     (`cuda_host_register` → `hipHostRegister`, already compiled, and
     `cuda_host_get_device_pointer` already handles WDDM host aliasing), plus verifying the
     `moe_handoff_amdgpu.cu` worker compiles and runs.
   - Budget here: only ~19 GiB of host RAM is free, so this can hold roughly 15-20 GiB of experts —
     enough to shave GLM (~8 GiB) or MiMo (~15 GiB) under the ceiling, not enough to be a general
     "run anything" tier.
2. **N-gram table on SSD** — already used for Qwen (24.44 GiB). Rows are gathered per forward with
   run-coalesced positioned reads. There is a full **Windows implementation already**
   (`exllamav3_ext/ngram_gather_win.cpp`, overlapped `ReadFile`, plus the Windows-only
   `EXL3_NGRAM_GATHER_PROF`) and the module releases the loader's file handles early on Windows.
   This is verification work, not porting.
3. **Tiered expert store / SSD** (`modules/expert_store.py`, `O_DIRECT` + pinned staging ring) —
   the DeepSeek-V4.1 mechanism. Out of scope for v1; it is where `O_DIRECT` → `FILE_FLAG_NO_BUFFERING`
   would be ported.

There is no llama.cpp-style whole-layer offload: attention, embeddings, norms and the lm head must
be on the GPU. For a dense model there is nothing to offload.

## 7. Risk gates

| Gate | Question | Status |
|---|---|---|
| **A. Memory** | is there room for the target pack + context? | **closed**: ~100 GiB usable, Qwen is 64.4 GiB resident |
| **B. Build** | does torch's HIP `cpp_extension` build on Windows? | **closed** with two fixes (section 8) |
| **C. Kernels** | do the 71 ROCm TUs compile and run correctly on Windows? | open — Phase 1/3 |
| **D. Attention** | is Triton available? | **closed**: triton-windows, AMD backend active |
| **E. Offload** | do the CPU-expert tier and ngram SSD streaming work on Windows? | ngram: code exists (verify); CPU tier: verify after Phase 1 |

## 8. Verified blockers and their fixes

Both were found by building a real extension on this box; both are required for `pip install`.

1. **torch's hipify loses every Windows path.** `torch/utils/hipify/hipify_python.py` builds
   `all_files` with backslashes, then compares against `_to_unix_path(filepath)`. Every source falls
   into the "ignored" branch, `hipified_path` becomes `None`, and `object_file_path(None)` raises
   `TypeError`. Fix (gated to Windows, in `setup.py` and `ext.py`): wrap
   `hipify_python.preprocessor` so `all_files` is normalized first. Two lines, no Linux change.
2. **Device library path.** Without it: `clang: error: cannot find ROCm device library`. Set
   `HIP_DEVICE_LIB_PATH=<sdk>\lib\llvm\amdgcn\bitcode` (same trap `tools/strix_halo/env.sh`
   documents as TRAP 4).

Anticipated next (from reading the code, not yet hit):

3. `setup.py` / `ext.py` add `cublas.lib` on Windows unconditionally — skip when `torch.version.hip`.
4. `-Wno-register`, `-include format`, `--gcc-install-dir` are Linux-only; do not pass them.
5. `ngram.cu` includes `<unistd.h>`; guard it.

## 9. Phases

Effort in engineer-days, one person, box available. Phases 1 and 2 run in parallel.

### Phase 0 — Environment + probes — done

Environment as in section 2; harness committed as `tools/windows/probe_env.py`
(torch/device, memory ladders, compute, hipcc/headers/bitcode, `--genco` → `hipModuleLoad` →
launch, `--json`). Re-run after any driver/toolchain change and paste the JSON here.

### Phase 1 — Build the extension (1-2 d)

1. `tools/windows/env.ps1`: import the MSVC environment (`cmd /c "vcvars64.bat && set"` → PowerShell
   env), set `ROCM_HOME`/`HIP_PATH` (TheRock tree or `_rocm_sdk_devel`), `HIP_DEVICE_LIB_PATH`,
   `PYTORCH_ROCM_ARCH=gfx1151`, `MAX_JOBS`, `PATH`, and a `--check` mode mirroring
   `tools/strix_halo/env.sh --check`.
2. `tools/windows/build.ps1` / `rebuild.ps1`: `pip install -e . --no-build-isolation` in a dedicated
   `.venv` (keeps `build/` for incremental rebuilds) + copy `exllamav3_ext*.pyd` to the repo root.
3. Build-config deltas (gated): hipify patch, skip `cublas.lib` for HIP, skip Linux-only flags, add
   the devel include dir, `/bigobj` for `bindings.cpp` if needed.
4. Fix compile/link errors per TU (`ngram.cu` `<unistd.h>`, `stloader.cpp`, `hadamard.cpp`,
   `bindings.cpp`, anything clang rejects for the `x86_64-pc-windows-msvc` target).
5. `util/arch_list.py`: `rocminfo` fallback → `torch.cuda.get_device_properties`; accept `gfx1151`
   from the device name.

DoD: `import exllamav3_ext` works, all 71 TUs compile, kernel inventory matches a Linux build.

### Phase 2 — Runtime plumbing (2-4 d, parallel with Phase 1)

1. `util/hip_lib.py`: find `torch/lib/amdhip64*.dll` (`amdhip64_7.dll`), fall back to the loaded
   module; drop `/proc/self/maps`.
2. `util/hip_compiler.py`: accept `hipcc.exe`/`.bat`, `$EXL3_HIPCC`, `ROCM_HOME`/`HIP_PATH`, the
   `_rocm_sdk_devel` wheel; keep the compiler hash in the cache name.
3. The 7 JIT builders: drop `--gcc-install-dir`, keep `--genco --offload-arch=gfx1151`, use
   `%LOCALAPPDATA%\exllamav3\cache`. (The JIT round trip already passes on this box.)
4. `util/memory.py`: no-op `malloc_trim` / `set_host_mmap_threshold` / `swap_in_process`; route the
   `/proc/meminfo` readers through `host_memory_available()`.
5. `util/shm.py`: return `None` on Windows; keep `EXL3_HOST_MEM_RESERVE_MB` meaningful.
6. `model/moe_cpu_host.py`: Windows arena branch (no memfd/hugetlb); verify `SharedMemory` +
   `hipHostRegister` + the `moe_handoff_amdgpu.cu` worker.
7. Symlinks: replace the 14 git symlinks with real files that import the shared implementation from
   `tools/strix_halo/`.
8. `serve.py` signal handling behind `if os.name != "nt"`; verify Ctrl+C shutdown.
9. Run the CPU test suite (114 modules) on Windows; fix platform failures; document skips.

DoD: `import exllamav3` works, JIT kernels compile+load, CPU suite green, Linux unchanged.

### Phase 3 — Engine bring-up (2 d)

1. Tiny EXL3 model (1-4 B) first: load, greedy decode, `tools/bench.py`.
2. Qwen pack from `C:\P2027\port-kyo\model`: `-c 32768`, then longer. Confirm the ngram table
   resolves to disk streaming (`EXL3_NGRAM_GATHER_PROF` for a per-gather report), confirm paged
   attention runs on the AMD Triton backend, chat + tool call + stream, restart timing.
3. Quantized cache (`--cache-bits 8`) — Triton `qdeq`/`qwrite`; verify against the fp16 cache on a
   short prompt (needed for very long context).
4. Only if needed for GLM/MiMo: CPU expert tier (`EXL3_MOE_CPU_OFFLOAD`/`_SPLIT`), measure decode.
5. `tools/windows/bench.ps1` wrapping `tools/bench.py --json`.

DoD: Qwen serves OpenAI-compatible traffic on Windows with sane output and recorded numbers.

### Phase 4 — Performance (3-6 d, iterative)

1. Baseline: prefill 2K/8K/32K/128K, decode across chat/prose/code, first-token latency, vs
   `doc/benchmarks.md`. Compute is at Linux parity, so gaps are kernels/attention/memory.
2. In order: paged-attention under the Windows AMD Triton backend, `hgemm` autotune cache on
   Windows, `EXL3_BLOCK_GRAPH=1` under WDDM, pinned staging in `stloader`/CPU tier, Windows power
   and clock settings.
3. Re-run the memory ladder with the model resident to fix the safe context ceiling.

DoD: numbers stable, `doc/benchmarks.md` gains a Windows column, slow paths documented.

### Phase 5 — Productise (1-2 d)

- `doc/install-windows.md` (PowerShell), `tools/windows/{env,build,serve}.ps1`, README Windows
  section, troubleshooting (HIP_DEVICE_LIB_PATH, memory split, Triton), optional compile-only
  `windows.yml` CI job.

## 10. Test and verification strategy

- **Correctness**: CPU suite on Windows; engine smoke (load → greedy decode → tool call → stream)
  per model; `/health`.
- **Kernel coverage**: the startup log reports kernels/compiler; diff against a Linux run. A silent
  torch fallback is a perf bug, not a crash.
- **Numerics**: same prompt/pack, greedy, Windows vs Linux reference. Divergence at long context is
  expected (different attention kernels); early divergence is a bug.
- **Perf**: fixed harness, same prompts, before/after each change.
- **No Linux regressions**: every port edit behind `os.name == "nt"`; run the CPU suite on Linux
  before merging.

## 11. Work order

```
Phase 0 (done) ──► Phase 1 build ──► Phase 3 tiny model ──► Phase 3 Qwen ──► Phase 4 perf
                        └──► Phase 2 plumbing (parallel)
```

## 12. Open decisions

1. Target pack: Qwen3.8-Flash-Next (fits with ~35 GiB of headroom). Agreed.
2. GLM/MiMo: not v1. Revisit with the CPU expert tier once Qwen runs.
3. Vulkan: rejected (section 5).
4. Dedicated `.venv` (Python 3.12, same wheel set as the working Studio venv). Agreed.

## 13. Measurements log

| Date | Gate | Probe | Result |
|---|---|---|---|
| 2025-10 | A | `mem_get_info` | 107.87 GiB total; Windows sees 31.6 GiB RAM (96 GiB UMA carve-out) |
| 2025-10 | A | single allocation ladder | ≤88 GiB ok, 96 GiB fails |
| 2025-10 | A | aggregate 4 GiB blocks | 100 GiB ok, 104 GiB fails |
| 2025-10 | A | `OCL_SET_SVM_SIZE=262144` | no change |
| 2025-10 | A | Qwen pack on disk | 88.82 GiB total, 24.44 GiB ngram (SSD), 64.4 GiB GPU-resident |
| 2025-10 | B | one-TU torch HIP extension | builds+runs, 79 s |
| 2025-10 | B | hipify on Windows | broken (path separators) — patched, see 8.1 |
| 2025-10 | D | triton-windows 3.7.1.post27 | AMD backend active; elementwise + dot kernels pass |
| 2025-10 | — | fp16/bf16 4096³ | 33.7 / 30.3 TFLOP/s |
