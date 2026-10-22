# Kyojin on Windows: port plan and status (AMD gfx1151 / Strix Halo)

Status: **serving the shipped Qwen3.8-Flash-Next pack on bare-metal Windows 11.** The HIP
extension builds from source (all 71 ROCm TUs), the engine loads the pack, and
`tools/qwen/serve.py` answers OpenAI-compatible traffic. Branch: `windows-port` (worktree at
`C:\P2027\port-kyo\kyojin-win`).

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

## 2. Status: what is proven on this box

| Thing | Result |
|---|---|
| GPU | `AMD Radeon(TM) 8060S Graphics`, `gfx1151`, 20 CU, warp 32, integrated, **107.87 GiB** visible |
| torch | `2.11.0+rocm7.13.0` (win_amd64, HIP 7.13.99004), Python 3.12.10, dedicated `.venv` in the worktree |
| HIP toolchain | matching `rocm-sdk-devel==7.13.0` wheel from `repo.amd.com/rocm/whl/gfx1151/`, extracted to `.venv\Lib\site-packages\_rocm_sdk_devel`: `hipcc.exe`, AMD clang 23, target `x86_64-pc-windows-msvc`, headers, import libs, bitcode |
| Host compiler | `amdclang-cl.exe` from the same tree (see 8.4 — `cl.exe` cannot parse the SDK's HIP headers) |
| MSVC | VS2022 BuildTools `14.44.35207` and VS18 BuildTools `14.51.36231` both work (imported via `vswhere`/`vcvars64.bat`) |
| Triton | `triton-windows 3.8.0.post29`, backends `['amd', 'nvidia']`, active backend = AMD `HIPDriver` |
| Extension | `exllamav3_ext.cp312-win_amd64.pyd`, 76.7 MB, 131 exported symbols, 10 s no-op rebuild |
| Compute | fp16 33.7 TFLOP/s, bf16 30.3 TFLOP/s at 4096³ (Linux reference ~31) — no Windows penalty |
| Model | Qwen3.8-Flash-Next pack: **63.1 GiB resident**, n-gram table in `trellis_disk` mode |
| Server | `/v1/models`, `/v1/chat/completions`, early `/health`, start-up 52 s warm / 501 s cold |

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

Linux-only code, and where it stands:

| Area | File | Status |
|---|---|---|
| HIP runtime discovery | `util/hip_lib.py` | **ported**: finds `amdhip64_7.dll` in the wheel's `_rocm_sdk_core\bin` |
| HIP compiler discovery | `util/hip_compiler.py` | **ported**: `hipcc.exe`/`.bat`, `rocm-sdk.exe`, wheel devel tree, JIT flags |
| JIT kernels (7 builders) | `modules/gr_fn/gr_mix_hip.py`, `modules/ple_fn/ple_hip.py`, `modules/attention_fn/qsa_prefill_hip.py`, `modules/dsv41_fn/*.py`, `vendor/fla/hip/*.py` | **working**: qsa_prefill, gr_mix, ple, gdn/kda compiled and loaded at run time (the `--gcc-install-dir` clause already self-disables when the dir is absent) |
| Host memory tuning | `util/memory.py` | **ported**: `malloc_trim` already off on HIP, `set_host_mmap_threshold` degrades, `swap_in_process` returns `skipped` on Windows |
| Shared memory | `util/shm.py` | not yet: returns no capacity figure on Windows; only the CPU expert tier reads it |
| MoE CPU tier | `model/moe_cpu_host.py` | untested on Windows (needs the arena branch: no memfd/hugetlb; `SharedMemory` + `hipHostRegister` should carry it) |
| SSD expert tier (v2) | `modules/expert_store.py` | out of scope (`O_DIRECT` → `FILE_FLAG_NO_BUFFERING`) |
| `/proc/meminfo` readers | `expert_store.py`, `ngram_embedding.py`, `dsv41/prefix_cache.py` | routed through `util.memory.host_memory_available()` (psutil fallback now installed) |
| Server | `tools/{qwen,glm,mimo}/serve.py`, `tools/startup_health.py` | **working**: `os.register_at_fork` guarded |
| Repo layout | 15 git symlinks | **replaced with real files** (`core.symlinks=false` left them as text stubs) |
| Scripts | `build.sh`, `tools/strix_halo/*.sh` | replaced by `tools/windows/{env,build,serve}.ps1` |

## 4. Memory model on this box (the real constraint)

The machine is split at boot:

| Pool | Size | Notes |
|---|---|---|
| BIOS UMA carve-out (GPU dedicated) | ~96 GiB | Windows only sees **31.6 GiB** of system RAM |
| GPU shared/GTT | ~12 GiB | borrowed from the Windows side |
| **GPU visible to HIP** | **107.87 GiB** | `mem_get_info` total, stable regardless of host load |
| GPU actually usable | **~100 GiB** | 100 GiB of 4 GiB blocks ok, 104 fails; a single block caps at 88 GiB |
| Host RAM usable for offload / ngram-in-RAM | ~18 GiB free of 31.6 | `EXL3_HOST_MEM_RESERVE_MB` default 2 GiB |

Consequence: the model budget is fixed by the BIOS setting, not by desktop bloat — closing apps
does not add GPU memory. Raising the carve-out starves Windows; lowering it starves the GPU.
Single allocations over 88 GiB fail even with 107.7 GiB free (`OCL_SET_SVM_SIZE=262144` does not
change it), but the engine allocates per tensor, so the 88 GiB single-block limit does not bind.

**Qwen3.8-Flash-Next pack (measured):**

| Part | Size | Resident on GPU? |
|---|---|---|
| `model-0000X-of-00009.safetensors` | 63.34 GiB | yes |
| n-gram embedding table | 24.44 GiB | **no** — streamed from SSD per forward |
| `draft2s.st` + `hc_gs_sidecar` | 1.04 GiB | draft head (GPU) |
| **total pack** | **88.82 GiB** (95.4 GB) | **63.1 GiB measured resident** |

The n-gram table goes to disk automatically here: `NGramEmbedding` holds it in RAM only when
`available >= table + 48 GiB` (24.4 + 48 vs ~18 GiB free), so `EXL3_NGRAM_STREAM` resolves to
disk without any flag. That leaves ~35 GiB of the ~100 GiB budget for KV cache, recurrent state
and activations.

| Pack | Size | Fit on ~100 GiB |
|---|---|---|
| Qwen3.8-Flash-Next | 88.82 GiB (63.1 resident) | **yes**, served at ctx 32768 |
| GLM-5.3-Flash | ~92.9 GiB, no ngram table | tight; likely needs CPU expert offload or less context |
| MiMo-V2.6-Flash | ~98.5 GiB | no; needs CPU expert offload |
| DeepSeek-V4.1-Flash | 172 GiB | out of scope (SSD tier) |

## 5. Why not Vulkan

Vulkan is not a port, it is a rewrite, and it does not buy the memory this box is missing.

- **Kernel surface**: the ROCm build alone is 60 GPU TUs / ~22k lines, 23 of them using AMD inline
  asm or intrinsics (`v_dot*`, `__builtin_amdgcn_*`, MFMA/WMMA), plus ~9k lines of Triton in
  `modules/attention_fn/`. All of it is HIP-native: `.hsaco` code objects loaded through
  hipModuleLoad, GCN-specific GEMV/GEMM kernels that are the whole point of this fork, HIP graph
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

Decision: no Vulkan backend.

## 6. CPU and SSD offload: what the engine actually has

The EXL3 format is neutral about placement; the engine ships three placing mechanisms.

1. **CPU MoE expert offload** — for routed experts only.
   - `-mcl/--moe_cpu_offload N`: the routed experts of the first N block-sparse MoE layers live in
     system RAM and are computed on the CPU.
   - `-mcs/--moe_cpu_split N`: the tail N experts of *every* eligible layer live on the CPU, so the
     CPU GEMMs overlap that layer's GPU expert compute.
   - Requirements: layer-split mode (the default), `mul1` codebook, K ≤ 8, uniform per-expert
     biases. The Qwen pack is `"codebook": "mul1"` ✓.
   - Implementation: a spawned worker process per model component owns the expert weights and a job
     ring in pinned shared memory (`model/moe_cpu_host.py`); the CPU kernels are AVX-512 VNNI/AVX2
     (`exllamav3_ext/cpu/moe_mul1.cpp`); the GPU publish/wait flag kernels have a ROCm port
     (`cpu/moe_handoff_amdgpu.cu`) and the bindings are registered for ROCm. `MoeCpuHost` only
     raises when those bindings are missing.
   - Windows status: **untested**. The pieces that need attention are the arena (no
     `memfd_create`/`MFD_HUGE_*`), the shared-memory capacity check (`util/shm.py`), and
     `hipHostRegister` + the WDDM device alias (`cuda_host_get_device_pointer` already documents
     that rule). Budget here: only ~18 GiB of host RAM is free, so ~15 GiB of experts at most.
2. **N-gram table on SSD** — used for Qwen (24.44 GiB). Rows are gathered per forward with
   run-coalesced positioned reads; the Windows implementation already exists
   (`exllamav3_ext/ngram_gather_win.cpp`, overlapped `ReadFile`, plus the Windows-only
   `EXL3_NGRAM_GATHER_PROF`). **Verified working**: the pack selects `trellis_disk` and prefill
   reads rows from disk at 338 tok/s.
3. **Tiered expert store / SSD** (`modules/expert_store.py`, `O_DIRECT` + pinned staging ring) —
   the DeepSeek-V4.1 mechanism. Out of scope for v1.

There is no llama.cpp-style whole-layer offload: attention, embeddings, norms and the lm head must
be on the GPU. For a dense model there is nothing to offload.

## 7. Risk gates

| Gate | Question | Status |
|---|---|---|
| **A. Memory** | is there room for the target pack + context? | **closed**: ~100 GiB usable, Qwen is 63.1 GiB resident |
| **B. Build** | does torch's HIP `cpp_extension` build on Windows? | **closed** |
| **C. Kernels** | do the 71 ROCm TUs compile and run correctly? | **closed**: all compile, extension loads, hgemm bit-exact, GEMV/JIT kernels used in a real forward |
| **D. Attention** | is Triton available? | **closed**: triton-windows, AMD backend active |
| **E. Offload** | do the CPU-expert tier and ngram SSD streaming work? | **partial**: ngram disk streaming verified; CPU tier untested |

## 8. Verified blockers and their fixes

Every item below was found by building or running here. All are Windows-only code paths; none of
them changes Linux behaviour.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `TypeError` in `object_file_path` before the first compile | torch's hipify builds its source set with backslashes, compares against `_to_unix_path(filepath)`, so every TU is "ignored" | `util/win_hip.py:patch_hipify()` normalizes the set (upstream-worthy torch bug) |
| 2 | `clang: cannot find ROCm device library` | wheels keep bitcode at `lib\llvm\amdgcn\bitcode` | `HIP_DEVICE_LIB_PATH` (`env.ps1`, plus `win_hip.maybe_set_device_lib_path()` for JIT shells) |
| 3 | `error C2988` / `C2059` in `amd_hip_vector_types.h` for every `.cpp` TU | torch compiles `.cpp` with `cl.exe`; that header declares `typename HIP_vector_base<T,n>::Native_vec_*` before the class is defined, which MSVC evaluates eagerly | build with `CXX=amdclang-cl.exe` (`env.ps1 -Build`): clang-cl accepts the distutils cflags verbatim and emits COFF for `link.exe` |
| 4 | same, but only after the first include | `-include format` reached `cl.exe`: `_host_cxx_has_format()` probes `CXX`/`c++`/`g++`, and a Strawberry Perl g++ on PATH answers yes | never pass it on Windows (`setup.py`) |
| 5 | `'int32_t': is not a member of 'std'` in the SDK's vector types | libstdc++ leaks `<cstdint>` transitively, MSVC's STL does not | `/FIcstdint` for Windows+HIP |
| 6 | `__device__ function 'isgreater' cannot overload __host__ __device__ function` | MSVC's `<cmath>` declares `constexpr isgreater/...`; clang promotes constexpr to implicit host+device, colliding with hipcc's forced `__device__` declarations | `-Xclang -fno-cuda-host-device-constexpr` in the extension build and, via `HIPCC_COMPILE_FLAGS_APPEND`, for every JIT kernel too |
| 7 | device code cannot call `ring_of()`/`dec_exact()`/`gemv_r_ring()`/`send()` after (6) | the flag also removes the implicit promotion those helpers relied on | explicit `__host__ __device__` on the constexpr helpers device code calls (`quant/exl3_dec.cu`) |
| 8 | `LNK2001` for `hipblas*`/`rocblas_*`, `__truncsfhf2` | `link.exe` resolves everything up front; ELF links leave them to libtorch's dependencies. compiler-rt is not linked implicitly when objects come from clang | name `hipblas`, `rocblas`, `clang_rt.builtins-x86_64` (`win_hip.win_link_libs()`, used by `setup.py` and `ext.py`) |
| 9 | `mkdir` undeclared in `hgemm.cu` | dense-GEMM tune cache used `HOME` and POSIX `mkdir` | Windows branch with `LOCALAPPDATA` + `_mkdir` (the cache is what makes a restart take 52 s instead of 501 s) |
| 10 | AVX2 intrinsics "require target feature 'avx'" in `cpu/moe_mul1.cpp` | target attributes were gated on `__GNUC__ && __linux__`, which clang-cl does not define | add `defined(__clang__) && defined(_WIN32)` |
| 11 | `hip_lib: no libamdhip64.so in ...` | Windows keeps `amdhip64_7.dll` in `_rocm_sdk_core\bin`, not `torch/lib`, and there is no `/proc/self/maps` | portable search + bare-name fallback (`util/hip_lib.py`) |
| 12 | `AttributeError: module 'os' has no attribute 'register_at_fork'` | POSIX-only API in `tools/startup_health.py` | guarded |
| 13 | `serve.py` fails to import `serve_ping`/`serve_reqlog` | 15 git symlinks are text stubs with `core.symlinks=false` | replaced with real files |

Anticipated but not yet needed: `/bigobj` for `bindings.cpp` (compiled fine), devel include dir for
`EXL3_ROCM_DEV_INCLUDE` (unset).

## 9. Phases

### Phase 0 — Environment + probes — done

`tools/windows/probe_env.py` (torch/device, memory ladders, compute, hipcc/headers/bitcode,
`--genco` → `hipModuleLoad` → launch, `--json`). Re-run after any driver/toolchain change and paste
the JSON into section 13.

### Phase 1 — Build the extension — done

`tools/windows/env.ps1` (runtime env; `-Build` adds the MSVC environment, `CXX=amdclang-cl.exe`,
`PYTORCH_ROCM_ARCH=gfx1151`, `MAX_JOBS`; `-Check` reports what the shell can actually do) and
`tools/windows/build.ps1` (`setup.py build_ext --inplace`, keeps `build/` incremental, copies the
`.pyd` to the repo root). All 71 TUs compile; link succeeds; no-op rebuild 10 s.

### Phase 2 — Runtime plumbing — done for the single-GPU, no-offload path

hip_lib, hip_compiler (incl. the JIT flag), memory no-ops, symlinks, `startup_health` guard, psutil.

Still open, all off the Qwen path:

- `util/shm.py` + `model/moe_cpu_host.py` arena for the CPU expert tier (GLM/MiMo),
- `modules/expert_store.py` (out of scope),
- `tools/windows/bench.ps1` wrapper around `tools/bench.py`.

### Phase 3 — Engine bring-up — done

`load_qwen.py` (greedy decode) and `tools/qwen/serve.py` both work on the shipped pack at
`-c 32768`: `/v1/models`, `/v1/chat/completions` (coherent text, `reasoning_content`, `usage`,
`timings`), streaming, vision tower loaded, GPU memory fully released on exit.

### Phase 4 — Performance (next)

Baseline to fill in against `doc/benchmarks.md`:

1. `tools/bench.py` across prefill 2K/8K/32K/128K and decode across chat/prose/code.
2. Long context: 65536, then the memory ceiling with the model resident (`mem_get_info` per step).
3. `--cache-bits 8` (int8 KV, Triton `qdeq`/`qwrite`) — needed for very long context.
4. Startup cost: the `EXL3_VERIFY_*` self-checks and `EXL3_WARM_*` warm-up run at load; quantify
   what turning them off costs in confidence vs seconds.
5. Paged attention on the Windows AMD Triton backend, `EXL3_BLOCK_GRAPH=1` under WDDM.

### Phase 5 — Productise

`doc/install-windows.md`, README Windows section, `tools/windows/*.ps1` polished, troubleshooting
(HIP_DEVICE_LIB_PATH, `CXX=amdclang-cl`, memory split, Triton), optional compile-only
`windows.yml` CI job.

## 10. Test and verification strategy

- **Correctness**: CPU suite on Windows; engine smoke (load → greedy decode → chat → stream) per
  model; `/health`.
- **Kernel coverage**: the startup log names the compiler and the kernels it builds (`[kyojin] HIP
  kernel compiler: ...`); a silent torch/Triton fallback is a perf bug, not a crash — the first
  Qwen run fell back to Triton for QSA prefill and cost 4.2 s instead of 0.4 s to first token.
- **Numerics**: same prompt/pack, greedy, Windows vs Linux reference. Divergence at long context is
  expected (different attention kernels); early divergence is a bug.
- **Perf**: fixed harness, same prompts, before/after each change.
- **No Linux regressions**: every port edit behind `os.name == "nt"`; run the CPU suite on Linux
  before merging.

## 11. Work order

```
Phase 0 ✔ ─► Phase 1 ✔ build ─► Phase 3 ✔ Qwen ─► Phase 4 perf ─► Phase 5 docs
                  └──► Phase 2 ✔ plumbing (offload tier still open)
```

## 12. Open decisions

1. Target pack: Qwen3.8-Flash-Next. Agreed, now running.
2. GLM/MiMo: not v1. They need the CPU expert tier, which needs the arena/shm port.
3. Vulkan: rejected (section 5).
4. Dedicated `.venv` (Python 3.12, same wheel set as the working Studio venv). Agreed: `.venv` in
   the worktree, `uv` + the AMD release index.
5. `tokenizers`/`safetensors` must be pinned to the versions the engine was tested with
   (`0.22.2`/`0.8.0`); unpinned resolution pulled `1.0.0rc3`/`0.9.0rc1`, and `tokenizers` 1.x no
   longer exports `models`.

## 13. Measurements log

| Date | Gate | Probe | Result |
|---|---|---|---|
| 2025-10 | A | `mem_get_info` | 107.87 GiB total; Windows sees 31.6 GiB RAM (96 GiB UMA carve-out) |
| 2025-10 | A | single allocation ladder | ≤88 GiB ok, 96 GiB fails |
| 2025-10 | A | aggregate 4 GiB blocks | 100 GiB ok, 104 GiB fails |
| 2025-10 | A | `OCL_SET_SVM_SIZE=262144` | no change |
| 2025-10 | A | Qwen pack on disk | 88.82 GiB total, 24.44 GiB ngram (SSD), 63.1 GiB measured resident |
| 2025-10 | B | one-TU torch HIP extension | builds+runs, 79 s (TheRock 7.15 tree) |
| 2025-10 | D | triton-windows | AMD backend active; elementwise + dot kernels pass |
| 2025-10 | — | fp16/bf16 4096³ | 33.7 / 30.3 TFLOP/s |
| 2025-10 | B/C | 71-TU extension, matching 7.13 wheel tree | builds, links, imports; 10 s no-op rebuild |
| 2025-10 | C | `hgemm` 1024x2048x1024 fp16 | max err 0.0 vs torch reference |
| 2025-10 | C | `exl3_gemv_supported(0)` | True (Linux gfx1151 reference: False) |
| 2025-10 | — | `load_qwen.py`, ctx 4096, 24 tok | load 41.5 s warm; TTFT 0.40 s; decode 24.7 tok/s |
| 2025-10 | — | `serve.py` ctx 32768, cold | start-up 501 s (867 dense-GEMM tune calls, 420.8 s) |
| 2025-10 | — | `serve.py` ctx 32768, warm | start-up **52 s**; prefill 100 tok 0.30 s (338 tok/s); decode **36 tok/s** (55 tok/s short runs) |
| 2025-10 | — | process exit | GPU memory fully released (107.72 GiB free) |
