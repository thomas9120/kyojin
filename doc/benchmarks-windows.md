# Benchmarks: Windows

The first published numbers for Kyojin running natively on Windows 11. Same machine, same pack,
same method as [benchmarks.md](benchmarks.md), so the two columns are comparable; the port itself is
described in [windows-port.md](windows-port.md), the setup in [install-windows.md](install-windows.md).

## Results

Machine: Ryzen AI MAX+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5X with a 96 GiB BIOS UMA
carve-out (Windows sees 31.6 GiB of RAM), ROCm 7.13 on Windows, torch 2.11.0+rocm7.13.0, Python
3.12, MSVC 14.51 + clang 23 (`amdclang-cl`), triton-windows 3.8.

Pack: `Qwen3.8-Flash-Next-EXL3` (88.82 GiB on disk; 63.1 GiB resident on the GPU, the 24.44 GiB
n-gram table streams from the SSD in `trellis_disk` mode). Server: `tools/qwen/serve.py -c 32768`,
default environment, speculative decoding on.

| Measurement | Windows | Linux (this repo, same GPU class) |
|---|---|---|
| Prefill, ~3.3K tokens (median of 3) | **1395 tok/s** | 1621 (8K) |
| Decode, prose, 128 tokens | **44.6 tok/s** | 47.5 |
| Decode, chat, 128 tokens | **42.4 tok/s** | 52.8 |
| Decode, code, 128 tokens | **43.8 tok/s** | 73.2 |

`tools/bench.py --reps 3`, greedy, thinking on, the standard prompt set. Linux column:
[benchmarks.md](benchmarks.md), which uses the same prompts and the same client-side method.

### Prefill and decode against context

`tools/windows/bench_context.py` (server-reported `prompt_ms`; 64 generated tokens per point up to
27K, 32 at the longer ones). `-c 65536`, fp16 K/V:

| Prompt tokens | Prefill tok/s | Decode tok/s | GPU memory |
|---|---|---|---|
| 1 823 | 1391 | 42.9 | |
| 3 532 | 1555 | 42.1 | |
| 6 952 | 1533 | 43.9 | |
| 13 792 | 1524 | 37.4 | |
| 27 472 | 1571 | 47.1 | 63.1 GiB (`-c 32768`) |
| 46 852 | 1530 | 23.2 | 70.8 GiB (`-c 65536`) |
| 51 868 | 1528 | 43.1 | |

`-c 131072 --cache-bits 8` (int8 K/V pages, the Triton `qdeq`/`qwrite` path), then `-c 262144`:

| Cache | Prompt tokens | Prefill tok/s | Decode tok/s | GPU memory |
|---|---|---|---|---|
| 131072 | 50 197 | 1537 | 47.7 | |
| 131072 | 100 317 | 1495 | 34.1 | 71.4 GiB |
| 262144 | 208 884 | **1446** | 20.8 | **73.7 GiB** |

Prefill is flat from 3.5K to 100K (1391 to 1495 tok/s, Linux: 1621 at 8K down to 1505 at 128K) and
decode does not decay with context, which is the shape the Linux tables show as well
([benchmarks.md](benchmarks.md), since the attention is the same Triton/HIP code). The 23.2 tok/s
point at 46.9K is a first-touch tier transition, not a trend: the same prompt size measures 43-48
tok/s on the next run.

**Memory**: the pack is 63.1 GiB resident at `-c 32768`, 70.8 GiB at `-c 65536` (fp16 K/V), 71.4 GiB
at `-c 131072` and 73.7 GiB at `-c 262144` (both `--cache-bits 8`) — out of about 100 GiB usable, so
256K context fits with ~26 GiB to spare. The full window is served here: 208 884 prompt tokens at
1446 tok/s, against 1399 tok/s at 256K on Linux. Decode at that depth is 20.8 tok/s (Linux 46.9), the
same verify-row cost as everywhere else - see [DECODE-GAP.md](../DECODE-GAP.md). Measure it with the Windows GPU performance counter
(`\GPU Process Memory(*)\Dedicated Usage`): torch's `mem_get_info` is **not** reliable across
processes on WDDM and reports the full 107.72 GiB free while another process holds 70 GiB.

## What the gap is

> **Correction (2026-10-23).** There is no platform decode gap. The Linux card figures are thinking
> off, 256 tokens, and the Windows figures above are thinking on, 128 tokens. Measured with the Linux
> method, Windows gives 35.6 / 39.9 / **52.1** tok/s (prose / chat / code, 4.27 tokens per round on
> code). The Linux "~32 ms per round" below was never measured: the Linux cost table in
> `tools/qwen/spec_policy.py` (`CYCLE_MS`) puts a 4-row round at ~70 ms, and Windows rounds land within
> 1-5 % of it. Details are in [DECODE-GAP.md](../DECODE-GAP.md) section 0. The rest of this section is
> the original analysis, kept for its measurements.

Prefill is at parity: 1536 tok/s at 27K against 1548 (32K) on Linux is inside run-to-run noise, and
the 100K point is 1495 against 1505. Decode is 5 % behind on prose, 20 % on chat and 40 % on code,
and the cause is **not** drafting, the SSD, or the host.

Measured on this box (all against the same server, `-c 32768`, greedy, 256 tokens):

| | Windows | Linux (same GPU) |
|---|---|---|
| Decode, no speculation (1 row/forward) | 28.3 tok/s | - |
| Decode, mix policy | 38-43 tok/s | 47.5 / 52.8 / 73.2 |
| Accepted / rejected draft tokens, code | 146 / 76 | same wording, same rule |
| Tokens per round | 2.33 | 2.33 |
| Time per round | **62 ms** | **~32 ms** |

The draft policy is host code and the acceptance counts come out identical, so both builds verify
the same tokens in the same order: the gap is entirely in the cost of a round. Round cost, split
with `QWSPEC_FIXD` (one draft per round instead of the dynamic 3):

| Round | Cost |
|---|---|
| 1 row, no drafter (plain decode) | 35.3 ms |
| 2 rows + 1 drafter step | 50.4 ms |
| 4-7 rows + ~3 drafter steps | 62 ms |

Timed at the call boundary inside the serving engine (`serve.QwenEngine`, sync around each call, so
the absolute numbers are a little high and the split is what counts):

| Phase | ms/round | calls/round |
|---|---|---|
| target verify forward | **61.8** | 1.03 |
| drafter layer forward | 9.0 | 3.50 |
| drafter sampling (mixer + head) | 3.8 | 3.32 |

The drafter costs ~13 ms a round; the verify forward costs three times that, and it is the forward
that scales with rows:

| Verify rows | ms/call |
|---|---|
| 1 (plain decode) | 35.3 |
| 2 | 43.2 |
| 3 | 48.5 |
| 4 | 54.1 |
| 5 | 62.9 |
| 6 | 66.2 |
| 7 | 76.1 |
| 8 | 76.9 |

**~5.5-6 ms per extra row**, flat across the range. That is the gap: a 4-row round costs the target
model 54 ms here, and on Linux the whole round - drafter and 4-row verify together - costs ~32 ms.
The extra rows should be nearly free (the weights are read once either way), and on Linux they are.

Ruled out as the cause of that per-row cost, each measured on the same server:

- **MoE decode kernel choice.** `EXL3_MOE_VALU=0` (per-expert path), `EXL3_MOE_VALU=0
  EXL3_DEC_MOE_UNION=1` (expert-union path) and the default all land within 0.1 ms/round of each
  other (62.6-62.9); `EXL3_MOE_FUSED=0` is worse (67.5). So it is not which expert kernel runs.
- **Kernel dispatch overhead.** An empty launch costs 4.62 us on this box (216 k launches/s),
  normal for HIP.
- **Memory residency.** The weights are in dedicated VRAM (WDDM adapter counters), and prefill
  streams the same weights at parity, so nothing is being read through a slow shared-memory path.
- **The n-gram table.** 2.5-4 ms per round (above).

What it is *not*:

- **GPU-bound, not host-bound.** `\GPU Engine(*)\Running Time` for the server process during a
  decode is 90 % of the wall time, so per-launch host overhead is not the limiter.
- **Block graphs are slower on Windows, not faster.** `EXL3_BLOCK_GRAPH=1` drops decode from 37.9
  to 28.1 tok/s (same tokens; the graph replay itself costs more here than the eager launches it
  replaces). Left off; the knob is documented but unhelpful on this platform.
- **The n-gram table is not the problem.** A decode round gathers 70-120 rows from the 24.44 GiB
  streamed table and spends 2.5-4 ms doing it: 4-6 % of a round, not 40 %. A row cache now sits in
  front of that path (`EXL3_NGRAM_CACHE_MB`, on by default on Windows at 1 GiB) and hits 30-35 % of
  the rows, which is worth ~1 ms a round - real, but it does not move the code number.

## Method notes

- `tools/bench.py` measures prefill as prompt tokens / wall time; `tools/windows/bench_context.py`
  reports the server's own `timings.prompt_ms` (prefill only, without the first generated token).
  The two agree within 4 % on this box once the client is not the bottleneck.
- Use `127.0.0.1`, never `localhost`: on Windows `localhost` resolves to `::1` first and each
  request stalls about 2 s before falling back to IPv4, which alone drops the reported prefill from
  1395 to 765 tok/s. `tools/bench.py` defaults to `127.0.0.1` since this run.
- First start after a build is not representative: the dense GEMM tuner runs 867 rocBLAS calls
  (420 s on this GPU) unless `%LOCALAPPDATA%\.cache\exllamav3\dense_gemm_tune.txt` exists. Both
  columns are steady state.
- Decode numbers come from the client's first-to-last streamed chunk, so they include the network
  round trip; loopback on this box is ~0.1 ms.

## Next

0. The per-row cost above is written up on its own, with the measurement recipes, the ruled-out list
   and the remaining suspects: [../DECODE-GAP.md](../DECODE-GAP.md).

1. **Why an extra verify row costs 5.5-6 ms.** That is the whole decode gap, and the four obvious
   explanations are measured and gone (see above). What is left is per-row work inside the forward
   whose cost does not amortize: the candidates are the recurrent/conv state updates of the 36
   linear-attention layers, the QSA sparse attention plan, and the hyperconnection mix sites (96 of
   them) - each of which is a 1-row op today and would be re-run per row. The next step is to time
   those three in isolation for R = 1..8 (a temporary sync around each in `transformer.forward`,
   the method used for the table above), then compare with the same measurement on Linux.
2. Contexts between 256K and the 262144 page-table limit are untested (the model's window is 256K).
3. If the drafter turns out to be fixed cost rather than wasted work, revisit speculation itself:
   fewer, longer drafts (or `QWSPEC_*` thresholds tuned for a slower drafter) may beat the shipped
   rule on this platform even though the rule is right on Linux.
