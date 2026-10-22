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

`-c 131072 --cache-bits 8` (int8 K/V pages, the Triton `qdeq`/`qwrite` path):

| Prompt tokens | Prefill tok/s | Decode tok/s | GPU memory |
|---|---|---|---|
| 50 197 | 1537 | 47.7 | |
| 100 317 | 1495 | 34.1 | 71.4 GiB |

Prefill is flat from 3.5K to 100K (1391 to 1495 tok/s, Linux: 1621 at 8K down to 1505 at 128K) and
decode does not decay with context, which is the shape the Linux tables show as well
([benchmarks.md](benchmarks.md), since the attention is the same Triton/HIP code). The 23.2 tok/s
point at 46.9K is a first-touch tier transition, not a trend: the same prompt size measures 43-48
tok/s on the next run.

**Memory**: the pack is 63.1 GiB resident at `-c 32768`, 70.8 GiB at `-c 65536` (fp16 K/V) and
71.4 GiB at `-c 131072` (`--cache-bits 8`) — out of about 100 GiB usable, so 128K context fits with
~29 GiB to spare. Measure it with the Windows GPU performance counter
(`\GPU Process Memory(*)\Dedicated Usage`): torch's `mem_get_info` is **not** reliable across
processes on WDDM and reports the full 107.72 GiB free while another process holds 70 GiB.

## What the gap is

Prefill is at parity: 1536 tok/s at 27K against 1548 (32K) on Linux is inside run-to-run noise.
Decode is 5 % behind on prose, 20 % on chat and 40 % on code.

Code is the interesting one. On Linux, code decode is much faster than prose (73.2 vs 47.5) because
source text has long confident n-gram chains: the draft model proposes many tokens per round and
the verification is nearly free. On Windows the same draft runs, but **every round reads its n-gram
rows from the SSD** — the table is 24.44 GiB and this box has ~18 GiB of host RAM, so
`NGramEmbedding` takes the `trellis_disk` path with no flag to change it. Code prompts touch a wide
spread of rows, prose and chat reuse a smaller working set that the OS page cache keeps hot.

That is the first thing to try for decode on Windows (see section "Next" below): a hot-row cache in
front of the disk path, or a smaller host-side reservation so part of the table can stay resident.

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

1. N-gram row cache for the disk mode (decode on code-heavy text is the one clear gap).
2. Contexts beyond 128K: 71.4 GiB at `-c 131072 --cache-bits 8` extrapolates to ~95 GiB at 262144,
   which should fit; untested.
3. The GPU is at 20 CUs (2 WGPs) — worth checking how much of the decode budget is the fused MoE
   kernel's grid barrier versus the GEMV path before tuning anything else.
