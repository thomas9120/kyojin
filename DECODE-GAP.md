# Decode gap on native Windows: a verify row costs 5.5-6 ms

Status: **closed: no platform gap in the verify forward.** The 5 % / 20 % / 40 % comparison came from
two different benchmark methods, plus a Linux round cost (~32 ms) that was inferred and never measured.
With the Linux method, and against the Linux cost model in the tree, a Windows round costs the same as
a Linux one to within 1-5 %. Section 0 has the evidence. Sections 1-8 are the original investigation,
kept for the recipes. Their claims about Linux are wrong where section 0 says so.

## 0. Resolution (2026-10-23)

### 0.1 The two columns were measured differently

- `doc/benchmarks.md:11`: the Linux card figures (47.5 / 52.8 / 73.2) are **thinking off, 256 tokens,
  "the four card prompts" per class**. Those prompts come from `prompts10.py` (imported by
  `tools/qwen/spec_bench.py`), which is not in this repository.
- The Windows column (`doc/benchmarks-windows.md`, and the runs in sections 1-3 below) is
  `tools/bench.py` with the server default, i.e. **thinking on (medium effort), 128 tokens, three
  prompts**. With thinking on, all 128 tokens of the "code" prompts are prose reasoning ("We need
  answer user: ... Need think through implementation ..."), so code gets prose-like acceptance.
  The table in section 1 shows exactly that: on Windows, code is no faster than prose.

Same engine, same pack, in-process (`serve.QwenEngine`, ctx 32768, the `tools/bench.py` prompts,
greedy, warm). Each cell is the mean of the 3 prompts:

| Method | prose | chat | code | tokens/round (prose / chat / code) | ms/round |
|---|---|---|---|---|---|
| thinking on, 128 tokens (what Windows published) | 44.2 | 42.8 | 43.4 | 2.96 / 2.89 / 3.06 | 64-78 |
| **thinking off, 256 tokens** (the Linux method) | **35.6** | **39.9** | **52.1** | 2.19 / 2.62 / **4.27** | 61-84 |

Tokens per round are **not** 2.33 everywhere. That figure was one 256-token run on Windows, and the
"Linux: 2.33" in section 1 was assumed (`drafting is host code`). It is not host code: the depth
rule reads the drafter's GPU-computed max probability every step (`spec_policy.install.sfs`).

### 0.2 The Linux round cost is in the tree, and Windows matches it

`tools/qwen/spec_policy.py:302` (`CYCLE_MS`) is the Linux measurement of a target forward for
1..8 rows: **30.3 / 41.6 / 50.2 / 58.6 / 68.8 / 75.1 / 84.7 / 92.9 ms**, plus 3.1 ms per drafted
depth and 2 ms of round overhead. That is ~9 ms per extra row on Linux (`README.strix-halo.md:113`
fits 32.4 + 9.04 x rows on an older build). Extra rows were never "nearly free" on Linux, and
the "~32 ms per round" in section 1 is impossible there: a 4-row Linux round costs 58.6 + 9.3 + 2 ≈
70 ms.

Windows rounds against that model, at each run's measured drafts per round:

| Run | drafts/round | Linux model | Windows measured | ratio |
|---|---|---|---|---|
| prose, thinking off | 2.00 | 58.4 ms | 61.3 ms | 1.05 |
| chat, thinking off | 2.33 | 62.2 ms | 65.4 ms | 1.05 |
| code, thinking off | 3.84 | 81.1 ms | 81.7 ms | 1.01 |
| prose, thinking on | 2.43 | 63.3 ms | 66.3 ms | 1.05 |
| chat, thinking on | 2.54 | 64.6 ms | 66.9 ms | 1.04 |
| code, thinking on | 2.80 | 67.6 ms | 69.7 ms | 1.03 |

Plain decode agrees as well: 27.3 tok/s on Linux (`tools/qwen/SERVE.md`, `-c 65536`), 28.3 on Windows.
The older Linux served A/B in the same file (thinking off, 160 tokens) gives prose / chat / code
38.7 / 42.2 / 52.2 tok/s. Windows with thinking off gives 35.6 / 39.9 / 52.1.

The 73.2 (and 66.8 / 67.0 on the second box) can only come from the unpublished `prompts10` code
prompts accepting more tokens per round: at the Linux cost model, 73 tok/s needs ~6 tokens/round.
The comparison can only be closed by running `tools/bench.py --thinking off --max-tokens 256` on
both platforms (option added for this), or by publishing `prompts10.py`.

### 0.3 Where the per-row cost goes (and why it is physics)

Per-module GPU time inside the verify forward (CUDA events, no syncs, the in-process harness of
section 7 with events instead of `synchronize`), ms per forward:

| | R=2 | R=4 | R=8 | per extra row |
|---|---|---|---|---|
| whole forward | 39.5 | 50.8 | 71.4 | 5.3 |
| fused MoE half-layer (block - attn - GDN - hc) | 16.2 | 27.2 | 38.4 | **3.7** |
| GatedDeltaNet (36 layers) | 9.2 | 10.4 | 14.3 | 0.85 |
| hyper-connection mix + apply | 6.0 | 4.7 | 7.4 | ~0.3-0.6 |
| full attention (12 layers) | 4.0 | 4.1 | 5.9 | 0.3 |

The MoE dominates. The fused kernel (`mf9`, available and in use on this build) already groups rows
by expert (`sorted_e` / `gstart`), so each distinct expert is read once per launch. A microbenchmark
of all 48 fused half-layers on synthetic residuals:

| R | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| identical rows (same experts) | 11.9 ms | 12.5 | 14.3 | 21.5 |
| independent rows | 11.8 ms | 19.1 | 30.6 | 52.6 |

An expert is 2.46 MB (K4 gate + up + down). Each independent row adds ~5.8 ms, which is 10 new experts
x 48 layers x 2.46 MB = 1.18 GB, i.e. **~203 GB/s**: the memory bandwidth of the box. Identical
rows cost almost nothing. In a real verify, rows share some experts, so they add ~3.7 ms each. That
is bandwidth-bound, it runs the same kernel on Linux, and no missing fast path can remove it.

### 0.4 The suspects of section 6

1. `exl3_dec_norm_route_r` is **not** an extension export on any platform. It is the Python method
   `BlockSparseMLP.dec_norm_route_r` (`exllamav3/modules/block_sparse_mlp.py`). On this pack it is
   bypassed anyway, because the fused MoE half-layer (`moe_fused.maybe_apply`) replaces norm, router
   and experts for R <= 8.
2. GDN: 0.85 ms/row, ~24 us per row per layer: real, small, same code on both platforms.
3. QSA / hyper-connections: 0.3-0.6 ms/row together.
4. "Compare against Linux with the identical harness": the Linux R table already exists (`CYCLE_MS`),
   and it is steeper than the Windows one.

Things that do not move acceptance (all 9 prompts, thinking on, token ids equal to the baseline):
`EXL3_DRAFT2S=0` (full draft head) and `EXL3_LAZY_DKV=0`. The drafter is not broken on Windows.

### 0.5 What would actually make decode faster (both platforms)

- Fewer expert bytes per verify row (more overlap, or lower-bit experts). The MoE is ~70 % of the
  per-row cost and is at the bandwidth limit.
- At R=1 the fused MoE runs at ~109 GB/s effective (11.9 ms for 1.30 GB). That is the one place
  with headroom, and it would help plain decode and every verify round alike.
- `doc/benchmarks-windows.md` should publish thinking-off numbers next to the thinking-on ones, so
  the two tables compare like with like.

Box: Ryzen AI Max+ 395, Radeon 8060S (gfx1151, 20 CUs, 107.87 GiB visible), 31.6 GiB host RAM with
a 96 GiB BIOS UMA carve-out, Windows 11, torch 2.11.0+rocm7.13.0, ROCm SDK 7.13 wheel. Pack:
`Qwen3.8-Flash-Next-EXL3` (48 layers, 512 experts, top-10, hidden 2560, moe_intermediate 640,
ngram 3/8 heads, ple_embed_dim 2560), served as `Qwen3.8-Flash-Yamz`. The Linux reference numbers in
`doc/benchmarks.md` were measured on the *same* hardware with ROCm, so every comparison below is
same-GPU.

---

## 1. The number to fix

| | Windows | Linux (same GPU) |
|---|---|---|
| Prefill, 8K / 32K / 128K | 1621-ish / 1536 / 1495 | 1621 / 1548 / 1505 |
| Spec decode, prose / chat / code | 42-47 / 42 / **38-42** | 47.5 / 52.8 / **73.2** |
| Plain decode (no speculation) | 28.3 | not published |
| Accepted / rejected draft tokens, 256-token code run | 146 / 76 | drafting is host code |
| Tokens per round | 2.33 | 2.33 |
| **Time per round** | **62 ms** | **~32 ms** |

The draft policy (`tools/qwen/spec_policy.py`) and the acceptance decisions are pure Python and
host-side, and the accepted/rejected counts come out identical run to run and platform to platform.
So both builds verify the same tokens at the same rate per round, and the entire gap is the cost of
one round. 32 ms vs 62 ms at 2.33 tokens/round is exactly 73 vs 38 tok/s.

## 2. Where the round goes

`torch.profiler` returns **no CUDA events at all** on this ROCm/Windows wheel (CPU aten ops only, and
only a handful), so attribution below is done by timing at the Python call boundary with
`torch.cuda.synchronize()` around each phase. That inflates the absolute numbers by a few ms per
round but preserves the split.

`tools/qwen/serve.QwenEngine` built in-process (ctx 4096), greedy, code prompt, 128 tokens, 38 rounds:

| Phase | ms/round | calls/round | ms/call |
|---|---|---|---|
| target verify forward (`model.forward`) | **61.8** | 1.03 | 60.2 |
| drafter layer forward (`draft_model.forward`) | 9.0 | 3.50 | 2.58 |
| drafter sampling (`draft_model.sample_from_state`) | 3.8 | 3.32 | 1.14 |
| sum | 74.6 | | |

Prose is the same shape (55.9 / 7.0 / 3.0). So the drafter is ~13 ms a round - worth having, but a
fifth of the cost - and the target's verify forward is three quarters of it.

## 3. The verify forward is linear in rows

Same in-process harness, grouping every `model.forward` call by its row count (R = the verify batch):

| R rows | 1 (plain decode) | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| ms/call | 35.3 | 43.2 | 48.5 | 54.1 | 62.9 | 66.2 | 76.1 | 76.9 |
| calls in the 128-token code run | (separate run) | 4 | 6 | 17 | 3 | 3 | 3 | 2 |

That is **~5.5-6 ms per extra row, flat from 2 to 8**. (One R=20 call in that run - a chunked
catch-up - took 186 ms, i.e. it follows the same slope.)

This is the whole gap. Verifying R rows reads the same expert weights as verifying one row when the
rows share experts and the kernel reuses them; that is why speculation pays on Linux (a whole 4-row
round, drafter included, costs ~32 ms there, roughly what a *single* 1-row forward costs here). Each
row costing +6 ms means the per-row work is not amortizing: at 48 layers that is ~0.125 ms per layer
per row, which is about the cost of reading one row's 10 experts (24 MB at ~200 GB/s = 0.12 ms). So
the shape of the cost looks like "each row streams its own experts" - but the expert-union kernel is
not what is running, or is not helping (see §5).

For calibration: prefill is 1536 tok/s = 0.65 ms/row at R=4096, i.e. the same forward amortizes
weights 10x better at large R. Nothing is wrong with the kernels themselves; the small-R regime is
where the cost per row explodes.

## 4. What is not the cause (each measured, with the knob)

| Hypothesis | Test | Result |
|---|---|---|
| Uneven drafting / different policy | accept/reject counts and tokens/round | identical to Linux (2.33) |
| Host launch overhead (WDDM) | empty-kernel microbenchmark: `x.add_(1)` 20 000x, sync once | **4.62 us/launch**, 216 k launches/s - normal for HIP, not WDDM-inflated |
| Host-bound rather than GPU-bound | `\GPU Engine(*)\Running Time` for the server pid during a decode | **90 % of wall time** busy |
| HIP graphs would fix it | `EXL3_BLOCK_GRAPH=1` | **slower**: 28.1 tok/s vs 37.9, same tokens. Opposite of the CUDA result. Keep it off |
| MoE expert kernel choice | `EXL3_MOE_VALU=0`; `EXL3_MOE_VALU=0 EXL3_DEC_MOE_UNION=1`; `EXL3_MOE_FUSED=0` | 62.6 / 62.9 / 67.5 ms per round vs 62.5 baseline. **No difference** (fused off is worse) |
| Weights not resident (WDDM shared memory) | adapter counters while loaded: dedicated vs shared usage | dedicated VRAM, and prefill streams the same weights at parity -> not a residency path |
| n-gram table streaming from SSD | per-stage profiling, `EXL3_NGRAM_PROF=1` | 2.5-4.2 ms per round (70-120 rows at ~37 us each). 4-6 %, **not** 40 % - an earlier write-up blamed this and was wrong |
| Drafter chain | phase timing above, `QWSPEC_FIXD=1` vs default | ~13 ms/round total; real but secondary |

## 5. Relevant code, and what looks suspicious

- `exllamav3/generator/generator.py:1104-1121` - the per-round wrapper: `draft_fn(results)` then
  `iterate_gen(results, draft_tokens)`, with `EXL3_SPEC_PROF=1` recording `t_step`, `produced`,
  `drafted`, `t_verify`, `t_dfwd`, `t_dsamp`, `t_post`, `R` into `generator.SPEC_PROF_REC`.
  **Note:** in this tree those records came out empty when driving `serve.QwenEngine` - `SPEC_PROF`
  was `True` and `SPEC_PROF_REC` stayed empty, so serve's driver loop does not pass through that
  append site. Worth fixing before the next round of profiling; it is the cheapest instrumentation
  in the tree (it even syncs at each phase boundary for you).
- `generator.py:1381` - `params["dflash_verify"] = draft_tokens is not None`. This marker is what
  unlocks the four R-row fast paths (`block_sparse_mlp.py`'s MoE union branch,
  `dec_norm_route_r`'s R>1 branch, `quant/exl3.py`'s `dec_gemv_r` branch, `sliding_attn.py`'s
  `qkv_gemv_r_multi` branch). If any of those is unavailable here, the verify silently runs the
  batch-1 path per row - which is exactly the shape of the 6 ms/row cost.
- `exllamav3/modules/block_sparse_mlp.py:1672-1690` - the union branch. Two gates matter:
  `not (_MOE_VALU and self.support_hip_grouped and bsz <= 8)` (with `EXL3_MOE_VALU=1` from
  `serve.SERVE_ENV`, the union is skipped for exactly our verify sizes) and the branch being
  **opt-in** (`EXL3_DEC_MOE_UNION=0` by default; upstream comment: "no clear speed win yet at R=1 or
  R=6-8, only R=2..5"). Forcing it changed nothing here, which is itself a clue: either it did not
  engage, or the cost is not expert-streaming.
- `exllamav3/modules/block_sparse_mlp.py:794` - `support_hip_grouped` requirements (hidden and
  intermediate 128-aligned, top-k in 6/8/10, `full_expert_layer`, ...) - true for this pack.
- **`exl3_dec_norm_route_r` is not exported by this extension** (`hasattr(ext,
  "exl3_dec_norm_route_r") == False`, while `exl3_dec_moe`, `exl3_dec_moe_union`,
  `exl3_dec_gemv_r_multi` and `exl3_moe_fused_*` are). If Linux exports it, that missing R>1 routing
  path is a prime suspect and cheap to check (compare `dir(exllamav3_ext)` between platforms).
- `exllamav3/architecture/qwen4_exp_mtp.py:167` - `sample_from_state`: per draft step it collapses
  the stream stack through the mixer, prepares lm_head input, then either the two-stage draft head
  (`draft2s`, when `bsz*seq == 1` and `EXL3_DRAFT2S` is set - it is) or the full 151 936-row head.
- Traps hit while measuring (do not repeat the mistake): `ext.exl3_gemv_supported(x)` takes a
  **device index**, not K. `exl3_gemv_supported(0)` is `True` on this build (gfx115x is WMMA family
  2); looping `K` over it returns False for every "device" 1..8 and reads like "GEMV unsupported".

## 6. Remaining suspects, ranked, each with its test

1. **The R>1 routing/norm path (`exl3_dec_norm_route_r`) is missing.** Test: `dir(exllamav3_ext)` on
   Linux vs Windows; if it is there, find why `build_config.py`'s ROCm source list drops it (or
   whether the export is `#if`'d out for non-gfx12) and build it.
2. **Per-row work in the 36 GDN (linear attention) layers** - recurrent state update and the conv
   window per row: `exllamav3/modules/gated_delta_net.py`, the `dflash_verify` multi-row entry point
   at `:1007`. Test: temporary `torch.cuda.synchronize()` timers around the GDN call in
   `transformer.forward`, grouped by R, same harness as §3.
3. **QSA sparse attention plan** (`attn.py:716`, `:772`; `EXL3_VERIFY_QSA_PROJ` /
   `EXL3_VERIFY_QSA_ATTN` force the per-row fallback) and the **96 hyperconnection mix sites**
   (`blocks/hyperconnections.py`, `gr_mix.hip` JIT kernels). Test: same grouped timers; and
   `EXL3_VERIFY_QSA_*=1` A/B to see whether the batched plan is even in use above the sparse
   threshold (2051 tokens - note the decode tests above ran at 4K context, so QSA *was* active).
4. **Compare against Linux with the identical harness.** `doc/benchmarks.md` numbers are end-to-end;
   what is missing is the R-table from §3 on Linux. One `prof_r.py` run there answers whether Linux
   also pays ~6 ms/row (in which case the gap is elsewhere, e.g. round count) or whether Linux's rows
   are nearly free (in which case a fast path is missing here).

## 7. Reproducing the measurements

All of these were run with the venv interpreter and, for the JIT builders, with
`tools/windows/env.ps1` dot-sourced (it sets `HIP_DEVICE_LIB_PATH` and `--rocm-path`; without them
the runtime JIT dies with "clang: error: cannot find ROCm device library").

**Phase split + R table** - build the real engine and wrap the three call sites (`$ROOT` = repo):

```python
import asyncio, functools, os, sys, time
from collections import defaultdict
ROOT = r"C:\P2027\port-kyo\kyojin-win"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "qwen"))
import serve                                  # before exllamav3: serve sets the serving env
eng = serve.QwenEngine(r"C:\P2027\port-kyo\model", ctx=4096)
torch = eng.torch
fwd, on = defaultdict(lambda: [0.0, 0]), {"v": False}

def wrap(obj, name, sink, rows_of):
    orig = getattr(obj, name)
    @functools.wraps(orig)
    def timed(*a, **k):
        if not on["v"]: return orig(*a, **k)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig(*a, **k)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        s = sink(rows_of(a, k)); s[0] += dt; s[1] += 1
        return out
    setattr(obj, name, timed)

wrap(eng.model, "forward", lambda R: fwd[R],
     lambda a, k: int((k.get("input_ids") if "input_ids" in k else a[0]).shape[-1]))
wrap(eng.draft_model, "forward", lambda R: fwd[-1], lambda a, k: 0)      # drafter rows are 1
wrap(eng.draft_model, "sample_from_state", lambda R: fwd[-2], lambda a, k: 0)

def run(prompt, n):
    async def go():
        st = {}
        async for _ in eng.generate(prompt, stats=st, max_tokens=n,
                                    sampling={"temperature": 0.0}, stop=[], speculative=True):
            pass
        return st
    return asyncio.run(go())

run("Write a Python function that parses an ISO-8601 duration string into seconds, with tests.", 24)
fwd.clear(); on["v"] = True
st = run("Write a Python function that parses an ISO-8601 duration string into seconds, with tests.", 128)
on["v"] = False
for R, (sec, n) in sorted(fwd.items()):
    print(R, n, f"{1000 * sec / n:.1f} ms/call")
```

**End-to-end A/B of an env knob** (starts a fresh server per arm, waits for `/health`, runs the same
body twice, then stops it):

```bash
# arm = "<ENV=VAL ...>" label ; uses tools/windows/serve.ps1 -Ctx 32768
env EXL3_MOE_VALU=0 nohup powershell -NoProfile -ExecutionPolicy Bypass \
  -File tools/windows/serve.ps1 -Ctx 32768 > /tmp/serve_arm.log 2>&1 &
# ... wait for "READY on", then: max_tokens 256, temperature 0, code and prose prompts,
# read timings.predicted_per_second, accepted_draft_tokens, rejected_draft_tokens
```

**GPU busy ratio during a decode** (PID of the listening python, engine counter, `Running Time` is
in 100 ns units):

```powershell
(Get-Counter '\GPU Engine(*)\Running Time').CounterSamples |
  Where-Object { $_.InstanceName -like "*pid_$PID_*" } |
  Measure-Object CookedValue -Sum   # sample twice, divide delta by 1e7 by the wall seconds
```

**Memory** must be read from the adapter counters, not torch: `torch.cuda.mem_get_info()` reports the
full 107.72 GiB free while the server holds 70 GiB (WDDM is per-process):

```powershell
(Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples |
  Where-Object { $_.CookedValue -gt 1e9 }
```

## 8. Practical notes that shaped the numbers

- Use `127.0.0.1`, never `localhost`: `::1` is tried first and each request stalls ~2 s before the
  IPv4 fallback. That alone reads as "prefill is 765 tok/s".
- Warm start ~50-68 s; a cold start after a build is 8 min because the dense GEMM tuner runs 867
  rocBLAS calls (cached in `%LOCALAPPDATA%\.cache\exllamav3\dense_gemm_tune.txt`).
- The first decode of a new row-count class pays the tune lookup; always warm up before measuring.
- `serve.ps1` does not forward `-mcl`/`--moe_cpu_offload` (use `EXL3_MOE_CPU_OFFLOAD`), and a stale server
  process can hold port 8000 - kill by command line:
  `Get-CimInstance Win32_Process | ? { $_.CommandLine -like '*serve.py*' }`.
- `EXL3_FUSED_SAMPLER` is forced to 0 by `ext.py:359` because the fused sampler kernel is not
  compiled into this extension, so every step downloads the 151 936 logits and samples on the host.
  Same on Linux-HIP, but it is ~0.1-0.3 ms per round of low-hanging fruit if it is ever ported.

## 9. Definition of done (superseded by section 0)

Decode at 32K context, code prompt, within 10 % of the Linux 73.2 tok/s - i.e. a verify round of
~35 ms at 2.33 tokens/round - without changing the tokens produced. `doc/benchmarks-windows.md` and
`doc/windows-port.md` both point here; update their tables when the slope is fixed.
