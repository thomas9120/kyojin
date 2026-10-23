# Code review — Kyojin inference engine (`windows-port`, commit `c802500`)

Reviewer: automated code review pass (pi). Date: 2026-10-23.
Repository state: branch `windows-port`, worktree `C:\P2027\port-kyo\kyojin-win`,
`HEAD = c802500` ("ci: drop the Linux jobs (Nix flake load, Ubuntu leg of the CPU tests)").

The Linux/Nix workflows being disabled is treated as intended (Windows-specific fork) and is **not**
counted as a finding. Section 6 discusses what the remaining CI does and does not cover.

---

## 1. Scope and method

Scope: the inference engine (`exllamav3/`, ~104 k lines of Python in 321 files plus the HIP/C++
extension), with emphasis on

* the fork's own engine changes (`git diff upstream/main...HEAD -- exllamav3/`: 10 files,
  `+405 / −22`),
* the core execution path a served request takes (Generator → Job → PageTable → Cache → modules),
* Windows-specific runtime behaviour (file I/O, `os.name` gating, build plumbing),
* the tests and CI that are supposed to protect those paths.

Method:

1. Diff review of the fork's engine changes against `upstream/main`.
2. Full read of the engine's central files (generator, job, pagetable, cache, sampler, spec-sample,
   n-gram embedding, safetensors loader, Windows HIP helpers) and of the code paths those call.
3. `pyflakes` over `exllamav3/` (found `cache/dsa.py:59` undefined `Config` annotation, duplicate
   `import time` in `generator/generator.py`, unused locals in `architecture/`, `conversion/`,
   `dsv41/`; nothing that is a live bug — details in F10).
4. Executed the repository's own test suite on this machine
   (`.venv` = Python 3.12.10, `torch 2.11.0+rocm7.13.0`, gfx1151, prebuilt
   `exllamav3_ext.cp312-win_amd64.pyd`). Numbers in section 5.
5. Targeted reproduction of individual failures to separate *engine* bugs from *environment/stale
   test* problems.

Not reviewed in depth: model architectures (`architecture/*`), the quantisation/conversion tooling,
the DS41 SSD-tier internals beyond portability, and the HTTP servers in `tools/` (they are outside
"inference engine"; only their CI role is mentioned).

---

## 2. Executive summary

| # | Severity | Area | Finding |
| --- | --- | --- | --- |
| F1 | **High** | `modules/ngram_embedding.py` | Host row cache publishes a slot in its index **before** the row data is written, and reads hit rows outside the lock. Concurrent staging (which the module explicitly supports) can return the wrong embedding row — silent numerical corruption, no error. |
| F2 | **High** (DS41 only), Low elsewhere | `modules/expert_store.py`, `dsv41/engram.py`, `modules/dsv41_fn/engram_io.py` | `os.O_DIRECT` is used unconditionally at **four** sites. On Windows the attribute does not exist, so the SSD expert tier and the Engram readers raise `AttributeError` instead of taking the documented buffered fallback. |
| F3 | Medium | `generator/generator.py:1530`, `generator/spec_sample.py` | Sampled speculative decoding computes the full sampling law for **every** verify position up front, even though a rejection at position *i* makes *i+1…R* irrelevant: O(R·V) memory, R full-vocabulary sorts + cumsums per round, and R wasted RNG draws. Trivially fixable by verifying lazily. |
| F4 | **Low** | `generator/spec_sample.py:107` | `draw()` clamps to the last vocabulary index, which can be a zero-mass token. When that token is also zero-mass in the target, `verify()` **rejects** it (`0 < 0`), so nothing is emitted; the degenerate acceptance (`q[d]==0`, `p[d]>0`) needs the last vocab index in the target's kept set but not the drafter's. Real but very rare — not the "emits a zero-probability token" failure the first pass described. |
| F5 | Medium | `ext_fallbacks.py:836-857` → `attention_fn/qwrite_triton.py:63` | The ROCm fallback for `quant_cache_paged` does not coerce `seq_len` to `int`, so a 0-dim tensor argument reaches Triton as a pointer and fails to compile. This is why 28 `test_kv_quant` cases fail here; it also means the 8-bit KV path has no working test on ROCm/Windows. |
| F6 | Medium | `modules/ngram_embedding.py:84-96, 581` | The 1 GiB host row-cache default is **per `NGramEmbedding` instance** (one per PLE layer, `ple_layer_ids` may list several), and the cache is created without the lock. Memory scales with layer count on a 128 GB box that is already tight. |
| F7 | Low | `generator/pagetable.py:566-577` | `next_evictable()` pops from the eviction deque without a bound check (`while available_pages[0]…`, then `popleft()`), turning "cache full" into an `IndexError`. |
| F8 | Low | `generator/generator.py:2091-2157` | `SpecGate` options use `x or default`, so an explicit `0`/`0.0` is ignored; `spec_pays()` divides by the EMA of the step time and raises `ZeroDivisionError` if a step is ever timed at exactly 0.0 s. |
| F9 | Low | `generator/job.py:236, 608` + `spec_sample.py` | Sampled drafting draws from the *same* `random.Random` as the target sampler, so enabling `EXL3_SPEC_SAMPLED` changes the RNG stream of every subsequent target token. Seeded A/B runs (spec on/off) are therefore not comparable. |
| F10 | Low | misc | Duplicate `import time` (`generator.py:4,33`), `Config` used in annotations without import (`cache/dsa.py:59`), per-token `next_token.item()` repetition, GPU logits retained per generated token when `return_logits`, lexicographic clang-version sort in `win_link_libs()`, one Win32 event object per positioned read, doc drift on `EXL3_PREFILL_BIG_CHUNK_MAXPOS`. |
| F12 | Low-Medium | `util/disk_store.py:175-196` | The prefix disk cache writes each entry through a fixed `<name>.pt.tmp` / `<name>.idx.tmp`, so two processes storing the same prefix interleave bytes before `os.replace`. Upstream already fixed exactly this in the image cache ("one temp file per writer", `3393ed9`) but not here. |
| F13 | Process | tests / CI | On this machine the suite reports **104 failed / 1137 passed / 461 skipped / 89 errors** (795 s cold, 333 s with warm Triton caches; identical counts) excluding `tests/ds41`, `tests/dsv41`. `tests/ds41/*` disable the GPU for the whole pytest process at import time, so a plain `pytest tests/` aborts at collection with 12 errors. CI runs 6 test files, none of which touch the engine. |

Only F1 and F6 live in code this fork actually added: `ngram_embedding.py` is the only file of that
set in `git diff upstream/main...HEAD -- exllamav3/` (the whole engine diff is 10 files, +405/−22).
F2, F5 and F12 are unmodified upstream code; they bite here because of the Windows/ROCm platform,
not because the port edited them. F3, F4 concern the sampled-drafting feature that the GLM server
ships enabled; F7–F10 are carried-over engine issues worth fixing while the tree is open; F13 is
about the review infrastructure, and matters because it is what would have caught F1 and F2.

**Verification pass.** Every finding was re-checked against this tree before any fix was made. F2,
F5 and the whole test/CI state (F13) were reproduced as commands, not just read; F1, F3, F6–F10 and
F12 were verified at the cited lines. The corrections below are applied in place: F4's consequence
was wrong (the zero-mass last token is rejected, not accepted), F2's fix list missed the two
`engram_io.py` sites, F5's C++ signature is `int seq_len` not `int64_t`, F10's `.item()` count is 2
per token, F8's division hazard is theoretical, F6's "key by table file" suggestion does not apply,
and the F13 breakdown undercounted by five subtest failures. No new findings.

---

## 3. Detailed findings

### F1 — Host row cache can serve a row that was never written (race)

**File:** `exllamav3/modules/ngram_embedding.py` (`_cache_plan` 598-613, `_cache_fill` 615-641,
`_gather_rows` 643-657).

```python
def _cache_plan(self, uids):
    ...
    with self._cache_lock:
        for i, u in enumerate(uids.tolist()):
            s = idx.get(u)                 # (a) index lookup under the lock
            ...
    return hit_pos, hit_slots, ...

def _cache_fill(self, miss_uids, rows):
    ...
    with self._cache_lock:
        for i, u in enumerate(uids):
            ...
            idx[u] = s                     # (b) publish the slot as valid ...
    if want_slots:
        c["rows"].index_copy_(0, ...)      # (c) ... but write the data outside the lock

def _gather_rows(self, uids, out):
    ...
    hit_pos, hit_slots, miss_pos, miss_uids = self._cache_plan(uids)
    if hit_pos.numel():
        out.index_copy_(0, hit_pos, self._cache["rows"].index_select(0, hit_slots))   # (d) outside the lock
```

Two threads can be inside `_gather_rows` at the same time: the prefetch worker
(`prefetch()` → `ThreadPoolExecutor.submit(self._stage, ...)`, line 713) and the calling thread
(`forward()` → `_stage(ids, pin)` in the `_match() is None` branch, line 756). The module's own
comment states this case is expected: *"forward() takes the set whose staged history equals the one
it was given and stages inline otherwise"* (`ngram_embedding.py:490-492`).

Interleaving that produces wrong data:

1. thread A misses on uid *u*, allocates slot *s*, publishes `idx[u] = s` (b), releases the lock;
2. before A reaches (c), thread B plans a hit on *u* and copies slot *s* into its staging buffer (d);
3. slot *s* either still holds the previous occupant of that slot (an evicted, different row) or, at
   start-up, uninitialised `torch.empty` memory.

No error is raised: the gather result is simply a different n-gram embedding row, so the output is
silently wrong. The same applies to an eviction that re-uses *s* between (b) and (c).

Secondary issues in the same block:

* `_cache_rows_init()` (581-596) is called without the lock (line 645), so two racing threads both
  allocate a full 1 GiB buffer and one is immediately dropped.
* `self._cache_stats["hit"|"miss"] += n` (647-655) and the `_cache_fill` counters are non-atomic;
  accounting can undercount under the same race (harmless, but it is the only observability for the
  cache).

**Why it matters:** the row cache sits in front of a *deterministic* lookup; handing back a
neighbouring row corrupts embeddings for the tokens whose n-gram hit the raced slot. It is exactly
the failure mode that is hardest to attribute in production (slightly degraded text, no traceback).

**Fix (small, no performance cost in the steady state):** make the plan→copy step atomic per call.
Either

* hold `self._cache_lock` across (d) and (c) as well — the critical sections are pure memory copies
  of ≤ `CACHE_MAX_ROWS` rows, and contention only occurs between a prefetch worker and an inline
  stage, which is rare; or
* publish validity separately: keep a per-slot "ready" flag/generation counter, set it under the
  lock *after* the `index_copy_`, and have `_cache_plan` re-check it before reporting a hit (with a
  fallback to a miss when it is not ready).

While there: take the lock in `_cache_rows_init`, and add a regression test that stages the same
uids from two threads with an injected sleep between (b) and (c) (the existing
`tests/test_ngram_row_cache.py` is single-threaded and cannot fail).

**Verification:** confirmed line by line — publish at `ngram_embedding.py:632`, data write at `:636`
outside the lock, unlocked hit copy at `:649`; the two-thread premise is the documented
`prefetch()`/inline-stage split (`:709-713` vs `:756`). No runtime reproducer: it needs the Qwen PLE
pack loaded, and `tests/test_ngram_row_cache.py` spawns no threads (it only creates the lock), so it
cannot fail this way. Severity stays High: the window is narrow and disk-mode only, but the result is
a wrong embedding row with no error and no trace.

---

### F2 — `os.O_DIRECT` unconditionally on Windows

**Files:** `exllamav3/modules/expert_store.py:958`, `exllamav3/dsv41/engram.py:233`,
`exllamav3/modules/dsv41_fn/engram_io.py:21` and `:135` (also `dsv41/engram.py:318` for the fallback
chain).

```python
# expert_store.py
fd = os.open(self.files[fid], os.O_RDONLY | os.O_DIRECT)

# engram.py
try:
    self.fds[path] = os.open(path, os.O_RDONLY | os.O_DIRECT)
except OSError:                      # <- AttributeError is not an OSError
    if not allow_buffered: raise
    ...
```

Reproduced here:

```text
tests/test_expert_store_protect_cpu.py::test_promote
E   AttributeError: module 'os' has no attribute 'O_DIRECT'. Did you mean: 'O_CREAT'?
    exllamav3\modules\expert_store.py:958
```

`os.O_DIRECT` is Linux-specific. On Windows, `os.O_DIRECT` raises `AttributeError` at *attribute
access*, which happens before `os.open()` is called, so

* `expert_store.ExpertStore._fd()` fails on the first promoted expert read;
* the same `except OSError` in `DirectRowReader` (`engram.py:233`), `ParallelRowReader`
  (`engram_io.py:21`) and `NativeRowReader` (`engram_io.py:135`) never triggers, and the
  `AttributeError` escapes the reader-chain fallback in `make_reader` (line 318 catches
  `ImportError, OSError, subprocess.SubprocessError`). The native/parallel readers are constructed
  *before* `DirectRowReader`, so fixing only `engram.py` leaves the DS41 path broken in a different
  place.

The DS41 SSD tier is declared out of scope for the Windows v1 in `doc/windows-port.md`, so this does
not contradict the published status — but the failure mode is a stack trace in the middle of a
generation rather than the documented "buffered" degradation, and the same code is compiled into
every Windows install.

**Verification:** reproduced exactly — `tests/test_expert_store_protect_cpu.py::test_promote`,
1 F, same traceback and line.

**Fix:** `flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)` at **all four** sites, keep `direct` a
real bool, and add `AttributeError` to the caught exceptions in `make_reader`. The `make_reader`
change alone is not sufficient: `direct` re-raises at the end of the chain, and the native/parallel
readers are constructed first. On Windows additionally consider `FILE_FLAG_NO_BUFFERING` via the
existing `_win_open_stream()` in `loader/safetensors.py`, which already does positioned, aligned
reads correctly; then the DS41 path is either supported or fails with an explicit "not supported on
Windows" error at construction time.

---

### F3 — Sampled verify computes the whole window eagerly

**Files:** `exllamav3/generator/generator.py:1527-1531`, `exllamav3/generator/spec_sample.py:38-75`.

```python
spec = [(spec_sample.verify(job_logits[0, pi, :], sr[2][pi], int(draft_tokens[j, pi]), sr[1], job.rng),
         None, None, None) for pi in range(npos - 1)]
```

`spec_sample.law()` builds a dense `float32` tensor over the padded vocabulary (≈200 k entries),
`fused_law()` sorts it, `draw()` cumsums it and `(p - q).clamp_(min=0)` allocates again — i.e. 5–8
full-vocabulary kernels plus a sort **per position**, for all `R-1 = npos - 1` positions, launched
before the first acceptance decision. If position 0 is rejected (the common case for a mediocre
drafter) every other law was pure waste, on GPU *and* in RNG draws.

`sr[2]` (the list of draft laws `q`) is kept on the GPU for the whole round as well, so the round
also holds `R` extra vocabulary-sized tensors.

**Impact:** decode rounds in the GLM server, where `EXL3_SPEC_SAMPLED=1`
(`doc/env_vars.md:1026`). The per-round overhead is roughly `(R-1) × (sort(200k) + ~6×200k)`. On this
iGPU that is very likely several milliseconds per round against a 20–35 ms decode step — i.e. a
measurable fraction of the speed-up the feature provides. Worth instrumenting before optimising
(`SPEC_PROF` already records `t_verify`).

**Fix:** move the verify into the accept loop and only evaluate position *i* after position *i-1* was
accepted (the loop already has exactly that structure — `reject_remainder()` breaks out on a
mismatch). That removes the wasted laws, the wasted RNG draws, and the `sr[2]` retention. It also
makes the "greedy vs sampled" branches structurally identical, which is easier to reason about than
the current pre-pass. The eager list also feeds the `_host_lean()` bulk-launch branch
(`generator.py:1533-1546`), but that branch is unreachable for sampled rounds: `_lean_verify_job()`
requires a single `SS_Argmax` sampler while `eligible()` only returns a law for
`SS_Sample`/`SS_Fused`. So the lazy rewrite loses no optimisation — the `spec[pi]` arm there is
already dead code. The per-position kernel count and RNG waste are confirmed by reading; the
millisecond estimate was not measured (as the first pass notes, instrument `SPEC_PROF` first).

---

### F4 — `draw()` can return a zero-mass last token (consequence corrected)

**File:** `exllamav3/generator/spec_sample.py:103-121`.

```python
def draw(p, u):
    cs = torch.cumsum(p, 0)
    idx = torch.searchsorted(cs, (cs[-1:] * u), right = True)
    return idx.clamp(max = p.numel() - 1)

def verify(logits, q, d, prm, rng):
    p = law(logits, prm)
    u1, u2 = rng.random(), rng.random()
    keep = u1 * q[d] < p[d]
```

If `cs[-1] * u` rounds up to `cs[-1]` (u within one ULP of 1.0) `searchsorted(..., right=True)`
finds nothing greater than the value and the `clamp` picks `numel - 1`.

**Corrected in the verification pass.** The first draft claimed this "usually" emits a
zero-probability token. It does not. In the usual case — a padded index, or a token outside the
target's top-K/top-P keep set — the target law is zero there too, so `verify()` computes
`0 < 0` → **rejected** and the residual draw runs. (Confirmed numerically with float32 laws: target
mass 0 at `d` gives `keep=False`; only `p[d] > 0` gives `keep=True`.) The degenerate acceptance needs
`q[d] == 0` **and** `p[d] > 0`, i.e. the last vocabulary index in the target's kept set but not the
drafter's — a real distributional bug, but far narrower, and the emitted token still has `p[d] > 0`.
Severity is therefore **Low**: this is two-line hardening, not a hot fix.

The probability of the clamp firing at all is still tiny (u must round to exactly `cs[-1]`; ~1e-7 per
draw), and the fix is two lines:

```python
u = min(max(u, 0.0), 1.0 - 1e-7)
...
keep = (q[d] > 0) & (u1 * q[d] < p[d])
```

and, for `q[d] == 0`, fall back to the residual draw (or treat it as a rejection).

---

### F5 — ROCm fallback for `quant_cache_paged` does not coerce `seq_len`

**Files:** `exllamav3/ext_fallbacks.py:836-857`, `exllamav3/modules/attention_fn/qwrite_triton.py:63-76`.

```python
def quant_cache_paged8(k_in, ..., cache_seqlens, block_table, seq_len, in_contiguous):
    _quant_cache_paged8_kernel[(bsz * seq_len,)](
        ..., seq_len, block_table.shape[1], in_contig=bool(in_contiguous), G=G, num_warps=1)
```

The CUDA build has `cache/q_cache.cu` compiled in, whose `int seq_len` parameter (`q_cache.cuh:48`)
is coerced by the pybind native-int caster, so a 0-dim tensor works. On ROCm that translation unit
is not part of the build (`build_config.py`), so the Python fallback runs instead and passes
`seq_len` through to Triton, where a 0-dim tensor becomes a `pointer<int32>`:

```text
tests/test_kv_quant.py::test_kv_quant[...]
E  triton.compiler.errors.CompilationError: at 6:8:
E      def _quant_cache_paged8_kernel(...): row = tl.program_id(0); b = row // seq_len
E      IncompatibleTypeErrorImpl('invalid operands of type pointer<int32> and triton.language.int32')
28 failed, 32 passed
```

The test passes such a tensor at `tests/test_kv_quant.py:114,132` (`q(new_cache_seqlens.amax())`),
which is the argument shape that fails. The verification pass reproduced the block above exactly
(`28 failed, 32 passed`, same Triton error text).

The engine's own call sites (`cache/quant.py:83,110`) pass a Python `int`, so served traffic is not
affected — but the CUDA and ROCm implementations of the same public extension function now disagree
about their accepted argument types, and the 8-bit KV cache (needed by Qwen past 262 K,
`README.md`) has *no* passing test on this port.

**Fix:** `seq_len = int(seq_len)` at the top of `quant_cache_paged8` (and the same for
`page_size`/`compand_a` in the torch fallback if they are used numerically). That alone turns the 28
failures into passes and restores cross-backend contract parity.

---

### F6 — Row-cache memory budget is per module instance, and its allocation is unlocked

**File:** `exllamav3/modules/ngram_embedding.py:84-96` and `581-596`.

```python
def _cache_mib() -> int:
    ...
    return 1024 if os.name == "nt" else 0
```

The value is a module-level constant, but `_cache` is an *instance* attribute allocated lazily per
`NGramEmbedding`. PLE models instantiate one `NGramEmbedding` per entry of `ple_layer_ids`
(`architecture/qwen4_exp.py:253-270`, `modules/ple.py:161`), and `ple_layer_ids` is a list. On a
model with k PLE layers the default becomes **k GiB** of host RAM, not 1 GiB. On the 128 GB box this
fork targets, with the model at ~63–74 GiB resident plus the streaming n-gram table, that is a real
budget item rather than a rounding error.

Two further observations:

* the budget is not shared with `check_host_memory()` (used for the RAM-mode table at line ~300), so
  nothing warns when the cache allocation pushes host memory to the edge;
* `CACHE_MAX_ROWS = 4096` means prefill chunks (64 k rows) bypass the cache entirely — intended per
  the comment, and fine, but it also means the *first* decode rounds after a prefill must re-read
  their rows; worth knowing when interpreting the profiling numbers.

**Fix:** allocate the cache from a process-wide budget (e.g. a small module-level
`_CACHE_BUDGET = [CACHE_MIB]` pool consumed by each instance), and log the total. Also take
`_cache_lock` in `_cache_rows_init`. Keying the cache by table file does **not** help here: each PLE
layer has its own key (`<key>.layers.N.ple.ple_embedding...`), so the layers never share a table.
Note too that `torch.empty` is lazy, so the unlocked-init race is a transient double *reservation*
(and eventual double commit), not two fully resident 1 GiB buffers.

---

### F7 — Unguarded `popleft()` in `next_evictable()`

**File:** `exllamav3/generator/pagetable.py:566-577`.

```python
else:
    while available_pages[0].ref_count:
        available_pages.popleft()
page = available_pages.popleft()
```

`build_eviction_order()` emits every unreferenced page at build time; pages claimed *by hash* later
in the same allocation loop stay in the deque but with `ref_count > 0`, which the `while` skips.
Once the deque is exhausted the `while` raises `IndexError` (deque index out of range) or the final
`popleft()` raises `IndexError: pop from an empty deque`. `iterate_start_jobs()` guards with
`job.current_new_pages_required() > self.pagetable.num_unreferenced_pages()` (which is conservative),
so this needs a specific interaction to fire — but the failure would surface as a raw `IndexError`
from the middle of `allocate_pages()` rather than the intended "not enough cache" path, and
`reap_failed_job()` would report it as a job error.

**Fix:** `while available_pages and available_pages[0].ref_count:` plus an explicit
`if not available_pages: raise RuntimeError("page table exhausted during allocation")` (or fall back
to `build_eviction_order()` again). Cheap and makes the invariant explicit.

---

### F8 — `SpecGate` falsy defaults and a division hazard

**File:** `exllamav3/generator/generator.py:2091-2157`.

```python
self.alpha = alpha or float(env("EXL3_SPEC_GATE_ALPHA", "0.3"))
self.probe_min = probe_min or int(env("EXL3_SPEC_GATE_PROBE", "4"))
...
def spec_pays(self):
    return self.tok_round / self.t_spec >= self.margin / self.t_plain
```

* `alpha=0.0`, `probe_min=0`, `margin=0.0` are silently replaced by the environment defaults; a
  caller cannot disable smoothing or set a zero-margin probe. `probe_max` uses the same `or` pattern.
* `spec_pays()` divides by `self.t_spec` / `self.t_plain`, which are EMAs of wall-clock
  `perf_counter()` deltas. A zero-length step makes `t_spec` exactly `0.0` and raises
  `ZeroDivisionError` inside `iterate()`, which the async wrapper treats as a fatal generator error.
  On the verification pass this turned out to be **theoretical**: `_iterate_gated()` only times the
  path that runs a full round, and two `perf_counter()` calls separated by a drafter/verify round do
  not return the same value on this clock. The one-line guard is still worth adding, but it is not
  a live crash.

**Fix:** `alpha if alpha is not None else ...`, and `max(self.t_spec, 1e-9)` in the comparison.

---

### F9 — Sampled drafting perturbs the target RNG stream

**Files:** `exllamav3/generator/job.py:235-237, 608`; `generator/spec_sample.py:110, 118-119`.

`Job.rng` is a single `random.Random`; `receive_logits()` seeds the sampler with
`self.rng.randint(0, 2**32-1)` and the sampled-draft path consumes `rng.random()` draws (two per
verified position, plus one per drafted token). Enabling `EXL3_SPEC_SAMPLED` therefore changes the
seed of every subsequent target token, so

* seeded A/B comparisons (spec on vs off, lookup on vs off) produce different token streams even
  when both are sampling from the same distribution;
* any future test that pins a seeded output across configurations will silently differ.

Not a correctness bug (each draw is independent), but it invalidates the natural "same seed, compare
outputs" debugging technique for the one feature where it is most needed.

**Fix:** derive a per-job RNG for drafting (`random.Random(hash((id(self), "spec")))` seeded from
the job seed) so target draws stay on the original stream.

---

### F12 — Prefix disk cache: one temp path shared by all writers

**File:** `exllamav3/util/disk_store.py:175-196` (`DiskStore._run`).

```python
tmp = self._p(name, ".pt.tmp")          # deterministic: <entry>.pt.tmp
...
torch.save(payload, tmp)
...
os.replace(tmp, self._p(name, ".pt"))
itmp = self._p(name, ".idx.tmp")        # deterministic too
torch.save(idx, itmp)
os.replace(itmp, self._p(name, ".idx"))
```

`name` is derived from the model identity and the prompt's page-hash chain, so two processes that
serve the same prompt (a server plus a benchmark run, two `serve.py` instances on one model, or the
same pack used from two worktrees) compute the *same* entry name and write the *same* temp files.
They can then interleave their bytes before `os.replace`, and the `.idx` of one writer can be
published over the `.pt` of the other — a torn entry that `read_idx()` accepts (the index is valid
JSON) and `read()` later unpickles into a corrupt tensor. The existence check at line 175
(`if os.path.exists(...idx): touch; continue`) narrows the window but does not close it: both
writers check before either replaces.

This is not hypothetical for this project: upstream hit the identical bug in the image disk cache and
fixed it in `fix/image-cache-tmp-per-writer` (`3393ed9`, "Two requests storing the same picture wrote
to the same `<name>.tmp`, so their bytes could interleave before `os.replace`. Each writer now uses
`<name>.<pid>.<thread>.tmp`"). The prefix store uses a single background thread per process, so the
per-writer suffix only needs the pid.

**Fix:** `tmp = self._p(name, f".pt.{os.getpid()}.tmp")` and the same for `.idx`, keeping the
`*.tmp` cleanup glob (line 108-113) as-is. Cheap, consistent with the already-shipped image-cache fix.

### F10 — Smaller items (verified, low impact)

| Item | Location | Note |
| --- | --- | --- |
| Duplicate import | `generator/generator.py:4` and `:33` (`import time`) | Harmless; suggests a merge artefact. |
| Undefined annotation | `cache/dsa.py:59` (`config: Config \| None`) with no import | Only works because of `from __future__ import annotations`; any runtime introspection (`typing.get_type_hints`) or type checker breaks. |
| Repeated `.item()` | `generator/job.py:846, 862` (`next_token.item()` 2×/token) | Cheap on a CPU tensor but redundant; `token = next_token.item()` is computed at 862 and the text lookup at 846 repeats it. The first pass said 3–4×; re-checking found two. |
| `held_logits` retention | `job.py:862` (`self.held_logits.append(logits[:1, :, :])`) | With `return_logits`, one GPU logits row per generated token is retained until the job ends (`max_new_tokens × vocab × 4 B`). Consider a cap or CPU offload. |
| Clang version sort | `util/win_hip.py:win_link_libs()` (`sorted(os.listdir(crt_root), reverse=True)`) | Lexicographic: `"9.x"` sorts above `"22.x"`. Pick the highest numeric version. |
| Event object per read | `loader/safetensors.py:_win_pread_into` (`CreateEventW` per call) | Fine for reference reads; would be expensive if it ever became the hot path. The hot path is the C++ `ngram_gather_win.cpp`, so this is a note, not a defect. |
| Bare DLL name fallback | `util/hip_lib.py:find_hip_runtime` (`path = found[0] if found else _NAME`) | `ctypes.CDLL("amdhip64")` binds whatever the loader finds first; acceptable given torch registers the directory, but log the resolved module. |
| Doc drift | `doc/env_vars.md:993` says the servers set `EXL3_PREFILL_BIG_CHUNK_MAXPOS=32768`; `tools/glm/serve.py:70` sets `"0"` and the engine default is `0` (`job.py:1281`) | This table is what reproduces the published benchmark configuration, so a stale value is worth correcting. |
| Inconsistent attribute guards | `job.py:1935` (`g.tail_ckpt_max`) vs `prefix_disk.py:344` / `slot_store.py:223` (`getattr(g, "tail_ckpt_max", 0)`) | The attribute is always set by `Generator.__init__` (line 311), so the direct read is safe for real generators, but the mixed style is what lets stub-based tests drift; pick one convention. |
| Sampler mutates its input | `generator/sampler/custom.py:1181` (`logits[..., actual_vocab_size:] = -inf`) | `Sampler.forward` masks the padded vocabulary **in place in the caller's tensor** (undocumented). It is currently consistent with what `return_logits` hands back, but any second consumer of the same logits (a re-sample, a scoring pass, an extension) sees the mutation. Consider documenting it or working on a view/copy. |
| pyflakes noise | `architecture/*` (f-strings without placeholders), unused locals in `conversion/`, `dsv41/` | Cosmetic; a `ruff`/`pyflakes` gate in CI would keep it from growing. Not re-run in the verification pass (`pyflakes` is not in `.venv`); the named examples were checked by inspection. |

---

## 4. Windows-port assessment (what looks right)

The port itself is careful and, in the parts I could check, correct:

* **Platform gating is disciplined.** Every fork change I read is guarded by `os.name == "nt"` (and
  where relevant `torch.version.hip`), so Linux behaviour is preserved — consistent with the hard
  rule stated in `doc/windows-port.md`.
* **The POSIX→Win32 file I/O rewrite is sound.** `loader/safetensors.py:191-266` models
  `CreateFileW`/`ReadFile`/`GetOverlappedResult` correctly (`argtypes`/`restype` for every call,
  `HANDLE` as `c_ssize_t` so `INVALID_HANDLE_VALUE == -1` compares, per-read `OVERLAPPED` offsets,
  `ERROR_IO_PENDING = 997` handled, event closed in `finally`). The distinction between the
  persistent unbuffered streaming handle and transient buffered reference handles
  (`_ref_fd`, `loader/safetensors.py:308-321`) is exactly the kind of detail that usually gets
  missed, and the reason is documented.
* **Build plumbing.** `util/win_hip.py` (`patch_hipify`, `HIP_DEVICE_LIB_PATH`,
  `win_link_libs`) and `util/hip_compiler.py` (`_hipcc_in`, `_win_jit_flags`) address real
  torch/ROCm-wheel problems rather than papering over them, and both are no-ops off Windows. The
  explicit `__host__ __device__` annotations in `exl3_dec.cu` are the correct fix for
  `-fno-cuda-host-device-constexpr`, not a workaround.
* **`hgemm.cu` cache path** (`LOCALAPPDATA` + `_mkdir`) is the right Windows analogue of
  `$HOME/.cache`, with a sane service-account fallback.
* **Line endings.** `.gitattributes` + `tools/windows/check_eol.py` in CI is a pragmatic guard for a
  tree that mixes bash tooling with Windows checkouts.
* **Documentation depth.** `doc/windows-port.md`, `doc/env_vars.md`, `doc/benchmarks-windows.md` are
  unusually detailed and, apart from the one stale row noted in F10, matched the code in the places I
  spot-checked.

The port's weak spot is not the Win32 code — it is the *concurrency* of the one feature added on top
of it (F1) and the absence of a Windows test gate that executes the engine (F13).

---

## 5. Test and CI state (measured, not estimated)

Environment: `.venv\Scripts\python.exe` = 3.12.10, `torch 2.11.0+rocm7.13.0`, one gfx1151 device,
prebuilt extension in the repo root.

```text
# plain run, as a developer would do it
python -m pytest -q tests/
13 skipped, 12 errors in 5.32s          # aborts during collection

# excluding the two directories that disable the GPU at import
python -m pytest -q tests/ --ignore=tests/ds41 --ignore=tests/dsv41 --continue-on-collection-errors
104 failed, 1137 passed, 461 skipped, 89 errors in 795.19s

# reproduced in the verification pass (warm Triton caches)
104 failed, 1137 passed, 461 skipped, 89 errors in 332.88s
```

Breakdown of the failures/errors (top files):

| Count | File | Cause (verified) |
| --- | --- | --- |
| 31 F | `test_hip_dsa_topk_gfx12.py` | gfx12-only kernel tests on gfx1151; not runnable here |
| 28 F | `test_kv_quant.py` | F5 (Triton arg type) |
| 21 F | `test_gated_delta_rule.py` | test calls `ext.cuda_recurrent_gated_delta_rule` with an outdated argument list → `TypeError` (stale test) |
| 15 E | `test_uncensor_bundled_cpu.py` | pytest `tmp_path` `PermissionError` (environment/basetemp) |
| 14 E | `test_bench_rocm_cpu.py` | same |
| 12 E | `test_hip_compiler_cpu.py` | same |
| 13 F | `test_multimodal_rewind_prefill.py` | 8 `FAILED` + 5 `SUBFAILED` (pytest-subtests), same cause: fake generator is a `SimpleNamespace` without `tail_ckpt_max` → `AttributeError` at `job.py:1935` (stale test; the real `Generator` sets it at `generator.py:311`) |
| 7 E each | `test_disk_store_cpu.py`, `test_moe_fused_k6_cpu.py`, … | environment (`tmp_path`, `/mnt/...` Linux paths) |
| 1 F + 7 E | `test_bench_gpu_target_cpu.py` | F: after the test re-mocks `kfd_gpu_target` to `""`, `gpu_name()` still reaches the real `torch_gpu()` fallback (the test assumes no GPU; passes on a CPU-only box). 7 E: `tmp_path` |
| 4 F | `test_glm_dec2.py` | fake module lacks `_dec_rowloop_rows` → `AttributeError` at `modules/mla_attn.py:648` (stale test) |
| 2 F + 5 E | `test_hip_lib.py` | the 2 F assert a POSIX path (`/venv/_rocm_sdk_core/lib/libamdhip64.so.7`); the Windows branch returns `amdhip64`. The 5 E are the `tmp_path` `PermissionError` |
| 2 F | `test_triton_paged_overflow.py` | test bug, not a kernel bug: `_empty_cuda_cache()` calls `torch.cuda.synchronize(device)` where `device` is the module's 0-arg lambda → `TypeError` *after* the assertions pass (stale test) |
| 1 F | `test_failure_containment.py` | `Generator.__new__` stub misses the new `deferred_jobs` attribute → `AttributeError` (stale test) |
| 1 F | `test_expert_store_protect_cpu.py` | F2 |
| 1 E | `test_ngram_prefetch_.py` | fixture model has no n-gram tensors → `load()` raises (stale test) |

The corrected failure rows account for all 104. Nearly all 89 errors are one local environment
problem, not the tree: pytest's basetemp is not writable here
(`C:\Users\pegas\AppData\Local\Temp\pytest-of-pegas` → `WinError 5`, 173 occurrences in the log).
Passing `--basetemp=<writable dir>` makes those tests run.

Two structural problems behind those numbers:

1. **Collection order determines whether the GPU exists.** `tests/ds41/test_alloc.py:15`,
   `test_ramfit.py:14`, `test_split.py:20` set `HIP_VISIBLE_DEVICES=-1` and
   `CUDA_VISIBLE_DEVICES=""` at *module import* time. pytest imports every module during collection,
   so by the time the GPU tests run, torch has no device: `test_ext_norm_.py` and others error with
   `RuntimeError: No CUDA GPUs are available`. Fix: move those assignments into a fixture (or a
   `pytest.mark.skipif` module-level `pytest.importorskip` pattern) and restore the environment
   afterwards; a bare `pytest tests/` should not be able to poison itself.
2. **The suite is not Windows-aware.** Hard-coded `/mnt/str/...` model paths (`test_qgemm.py:28`,
   `test_quant_fn.py:27`), POSIX assertions (`test_hip_lib.py`), and tests that need two GPUs
   (`test_device_copy_.py`) are run unconditionally. Most are one `pytest.mark.skipif(os.name !=
   "posix")` or a fixture-skip away from being useful again.

### What CI actually runs

`windows-cpu-tests.yml` (push/PR) runs:

* `python -m compileall -q exllamav3 tools tests` (syntax/import-time typos only),
* 6 test files: `tools/test_startup_health.py`, `tools/test_serve_logging.py`,
  `tools/qwen/test_guard.py`, `tools/qwen/test_spec_rule.py`, `tools/qwen/test_lookup_gate.py`,
  `tests/test_shared_tool_copies.py`,
* `python tools/windows/check_eol.py`.

`windows-rocm-build.yml` is a dispatch-only extension compile check; `build.yml` and
`container.yml` are `workflow_dispatch`/tag-triggered.

So **no CI job executes engine code** — not `Generator`, not `Job`, not `PageTable`, not the
`NGramEmbedding` cache. That is precisely why F1 and F2 are both invisible to CI, and why
`tests/test_ngram_row_cache.py` (the fork's own test for the new cache) is in the "excluded because
it imports exllamav3" comment at `windows-cpu-tests.yml:59-64`.

Minimum viable improvement, in order of value per effort:

1. `seq_len = int(seq_len)` (F5) + repair the stale tests listed above → the suite's pass/fail signal
   becomes meaningful again.
2. Fixture-based device masking for `tests/ds41` → `pytest tests/` stops aborting at collection.
3. A CPU-only job that runs the tests which do not need a model or a GPU by stubbing
   `exllamav3.exllamav3_ext` (`sys.modules` injection in `conftest.py`), or by importing the
   submodules directly instead of the package — enough to run `test_ngram_row_cache.py`,
   `test_pagetable`, `test_spec_sample`, `test_failure_containment`, `test_generator_max_new_tokens_`
   in seconds.
4. One nightly (or self-hosted gfx1151) job that runs the real engine tests; the fork's own docs
   already report hardware-specific numbers, so a machine exists.

---

## 6. Maintainability observations

* **Configuration surface.** 422 distinct `EXL3_*` variables are read from `exllamav3/**/*.py`
  (551 `os.environ` call sites) and `doc/env_vars.md` is 1058 lines. Several are read *per decode
  step* (`_host_lean()`, `_mtp_fuse_catchup()`, `EXL3_PREFILL_BIG_CHUNK` per chunk), which means
  behaviour can change mid-process and every measurement has to pin the environment. This is a
  deliberate A/B culture and it clearly produced the published numbers, but it is now the largest
  barrier to reviewing changes: any diff has to be reasoned about under `2^n` configurations. A
  read-once `settings` object (or at least caching the hot ones in module globals at import, as
  `fused_sampler_enable` already does) would not change behaviour and would make the hot paths
  reviewable.
* **Duplication in `tools/`.** `serve_reqlog.py` (628 lines), `serve_metrics.py`, `serve_ping.py`,
  `image_cache.py` exist three times (GLM/MiMo/Qwen) and `tests/test_shared_tool_copies.py` enforces
  byte equality. The rationale (symlinks do not survive Windows checkouts) is sound, and the test is
  the right mitigation — but three copies of a 628-line file is a maintenance tax on every log-format
  change. A `tools/_shared/` package with three thin re-export modules would keep the same runtime
  behaviour, keep the "one implementation" invariant structurally, and delete the test's reason to
  exist.
* **Assertion density in the engine.** `pagetable.py` uses `assert` for invariants that protect
  cache integrity (`assert self.ref_count > 0`, `assert self.phash not in ...`). Under `python -O`
  those vanish silently. If the tree is ever packaged or launched with `-O`, the page table loses its
  safety net; consider `if ...: raise RuntimeError(...)` in the page-table class (the hot decode path
  does not pay for it, since these run per page, not per token).
* **Comments are unusually good.** The dense "why" comments (e.g. the row-cache docstring's
  per-row-latency budget, the `MTP_FUSE_CATCHUP` trade-off notes, the `ngram_embedding` staging-set
  protocol) made this review tractable and are a real asset — they also make it easy to spot when the
  code has drifted from the comment (F1's "can only cost time" claim is exactly the case that is no
  longer true).

---

## 7. Suggested order of work

1. **F1** — make the row cache race-free (correctness, silent corruption, small diff, testable).
2. **F2** — `getattr(os, "O_DIRECT", 0)` at all four sites + catch `AttributeError` in `make_reader`
   (crash removal; DS41 users only, but the fix is a few lines).
3. **F5** — `int(seq_len)` in `quant_cache_paged8` (restores 28 tests; contract parity CUDA↔ROCm).
4. **F13 items 1–2** — repair stale tests, move the device masking into fixtures (makes the suite a
   usable signal again).
5. **F3** — verify lazily in the speculative loop (performance, GLM server; measure with
   `SPEC_PROF` before/after).
6. **F6, F7, F8** — small hardening; each is a few lines. F8's division guard is defensive (it needs
   a 0.0 s step, which this clock cannot produce).
7. **F12** — per-writer temp names in `DiskStore` (matches the fix already shipped in the image cache).
8. **F9, F10, F4** — cleanup and doc drift; F4 is the same two-line `q[d] > 0` guard, but with a much
   narrower impact than first reported, so it rides along with cleanup instead of driving priority.
9. **F13 item 3–4** — a CPU-only engine job in CI, then a nightly real-engine run.

---

## 8. Appendix — how to reproduce

```powershell
# fork diff reviewed
git -C C:\P2027\port-kyo\kyojin-win diff --stat upstream/main...HEAD -- exllamav3/

# static noise
uv tool run --from pyflakes pyflakes exllamav3 | Select-String -NotMatch "f-string|imported but unused"

# suite state (measured in this review)
.venv\Scripts\python.exe -m pytest -q tests/ `
  --ignore=tests/ds41 --ignore=tests/dsv41 --continue-on-collection-errors

# individual reproductions
.venv\Scripts\python.exe -m pytest -q tests/test_kv_quant.py                    # 28 failed, 32 passed (F5)
.venv\Scripts\python.exe -m pytest -q tests/test_expert_store_protect_cpu.py   # 1 failed, AttributeError (F2)
.venv\Scripts\python.exe -m pytest -q tests/test_ngram_row_cache.py                             # passes, single-threaded
.venv\Scripts\python.exe -m pytest -q tests/test_failure_containment.py                          # F13 stale test
```
