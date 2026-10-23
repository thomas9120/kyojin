# Deliberately untyped inside: `config`/`infer_params` arrive as loader kwargs, `handles` holds either
# a resident Tensor or a DiskTensorHandle (not being a Tensor is the whole point of disk mode), `ext`
# is swapped for a stub by the CPU tests, and every buffer (tables, _pins, _cache, ...) is None until
# load() fills it - so pyright's strict inference calls all of it Optional and reports ~90 findings in
# code whose runtime shape is fixed. These are the exact rules that noise comes from; the ones that
# catch typos, arity and protocol misuse stay on.
# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportCallIssue=false, reportIncompatibleMethodOverride=false

from __future__ import annotations
from typing_extensions import override
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import torch
from ..model.config import Config
from ..loader.safetensors import DiskTensorHandle
from ..ext import exllamav3_ext as ext
from . import Module
from .quant.exl3_lib.ngram_codec import ROW_DIM, mul1_codebook, dequant_rows, words_per_row


_MADV_POPULATE_READ = 22
_MADV_PAGEOUT = 21


def process_swap_bytes() -> int | None:
    """VmSwap of this process in bytes, None when unknown"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSwap:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _madvise_range(tensor: torch.Tensor, advice: int, byte_start: int = 0, byte_len: int | None = None, chunk: int = 1 << 30) -> int:
    """madvise a byte range of a CPU tensor in chunks of at most `chunk` bytes. Returns byte_len, -1 when madvise is unavailable or failed"""
    import ctypes
    libc = ctypes.CDLL(None, use_errno = True)
    page = os.sysconf("SC_PAGE_SIZE")
    total = tensor.numel() * tensor.element_size()
    byte_len = total - byte_start if byte_len is None else min(byte_len, total - byte_start)
    base = tensor.data_ptr() + byte_start
    lo = base - base % page
    hi = base + byte_len
    while lo < hi:
        n = min(chunk, hi - lo)
        if libc.madvise(ctypes.c_void_p(lo), ctypes.c_size_t(n), advice) != 0:
            return -1
        lo += n
    return byte_len

"""
Hashed n-gram embedding table (Qwen3.8-Flash-Next ple_embedding and kin): maps each token position
to (ngram_size - 1) * heads_per_ngram hash-table rows and concatenates them into one feature vector.

The table is enormous (tens of billions of parameters), so streaming is a first-class mode: by
default the table's tensors are never loaded — the module keeps DiskTensorHandles from the loader
and gathers only the rows a forward pass actually touches. The module is quantization-agnostic:

    <key>.trellis                  -> exl3_ngram_trellis format (util/convert_ngram.py)
    <key>.weight / .shard_N.weight -> unquantized source table

crossed with resident (RAM) or streamed (disk) storage gives four load modes, all returning
identical results for the same table contents.
"""


def _find_nth_prime_after(start: int, count: int) -> int:
    # mirrors the reference implementation used to derive the per-head vocab sizes
    def is_prime(v):
        if v < 2: return False
        if v % 2 == 0: return v == 2
        for d in range(3, int(v ** 0.5) + 1, 2):
            if v % d == 0: return False
        return True
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p):
            p += 1
    return p


PREFETCH_ENABLED = os.environ.get("EXL3_NGRAM_PREFETCH", "1") != "0"   # debug/A-B switch
PREFETCH_MIN_TOKENS = 256   # positions (bsz * seq) below which prefetch() declines (decode-sized)
MAX_PIN_SETS = 2            # staging sets: one with the last forward's uploads in flight, one being staged
PROF_ENABLED = os.environ.get("EXL3_NGRAM_PROF", "0") != "0"   # debug: print stage timings every 64 calls


def _cache_mib() -> int:
    """Host row cache in front of the disk path, in MiB; 0 when off.

    Every row read on the disk path is one unbuffered pread (FILE_FLAG_NO_BUFFERING), so it is a
    real device read even when the same row was read a second earlier: a decode round gathers
    ~70-120 rows and measured 2.6-4.2 ms doing it, i.e. ~40 us per row of SSD latency. Decode
    re-reads a small working set over and over (n-grams of natural text and especially of source
    code repeat), so a host cache of those rows removes most of that.

    Default: 1 GiB on Windows, where the table (24.4 GiB here) can never be resident and the
    budget is a *host RAM* one; 0 elsewhere, so Linux behaviour is unchanged. EXL3_NGRAM_CACHE_MB
    overrides either way (0 disables).

    The value is a *process-wide* budget: see _cache_budget_mib.
    """
    v = os.environ.get("EXL3_NGRAM_CACHE_MB")
    if v is not None:
        try:
            return max(0, int(v))
        except ValueError:
            return 0
    return 1024 if os.name == "nt" else 0


CACHE_MIB = _cache_mib()
CACHE_MAX_ROWS = 4096       # inputs larger than this bypass the cache (prefill chunks: 64k rows each)

# CACHE_MIB is a process-wide budget, not a per-instance one: a PLE model has one NGramEmbedding per
# entry of ple_layer_ids (architecture/qwen4_exp.py), so a per-instance default would be CACHE_MIB per
# layer, i.e. k GiB of host RAM for a k-layer PLE model. Each instance takes its share from this pool
# on first use (see _cache_rows_init) and gives it back in unload().
_cache_budget_mib = [CACHE_MIB]
_cache_budget_lock = threading.Lock()


def _cache_budget_reset():
    """Hand the whole row-cache budget back (tests build throwaway instances)."""
    with _cache_budget_lock:
        _cache_budget_mib[0] = CACHE_MIB


class _PinSet:
    """Pinned staging buffers for one hash + gather: unique row ids, the inverse map, the per-row
    head index and the packed rows. `held` while a queued prefetch or a running forward owns it;
    `event` marks the last forward's uploads from it."""

    def __init__(self):
        self.uids = None
        self.inverse = None
        self.heads = None
        self.packed = None
        self.event = None
        self.held = False

    def grow(self, n: int, row_words: int, row_dtype: torch.dtype):
        if self.uids is None or self.uids.numel() < n or self.packed.shape[1] != row_words \
                or self.packed.dtype != row_dtype:
            self.uids = torch.empty(n, dtype = torch.int64, pin_memory = True)
            self.inverse = torch.empty(n, dtype = torch.int64, pin_memory = True)
            self.heads = torch.empty(n, dtype = torch.int32, pin_memory = True)
            self.packed = torch.empty((n, row_words), dtype = row_dtype, pin_memory = True)


class NGramEmbedding(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        ngram_size: int,
        heads_per_ngram: int,
        ple_embed_dim: int,
        eos_token_id: int,
        stream_from_disk: bool | None = None,
        out_dtype: torch.dtype | None = torch.half,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        assert qmap is None, "NGramEmbedding quantizes via util/convert_ngram.py, not the qmap pipeline"

        self.ngram_size = ngram_size
        self.context_len = ngram_size - 1
        self.heads_per_ngram = heads_per_ngram
        self.num_heads = (ngram_size - 1) * heads_per_ngram
        self.ple_embed_dim = ple_embed_dim
        self.head_dim = ple_embed_dim // self.num_heads
        assert self.head_dim == ROW_DIM, f"expected {ROW_DIM}-D embedding rows, got {self.head_dim}"
        self.eos_token_id = eos_token_id
        # None: defer to config.infer_params.ngram_stream_from_disk at load time (the load-time
        # option; also EXL3_NGRAM_STREAM). An explicit bool here overrides it
        self.stream_from_disk = stream_from_disk
        self.out_dtype = out_dtype

        self.mode = None            # "trellis_disk" | "trellis_ram" | "fp16_disk" | "fp16_ram"
        self.K = None
        # The table is always kept as its individual shard tensors: RAM modes hold a list of CPU
        # tensors ((rows, 160) bf16/fp16 or (rows, words) int16), disk modes a list of
        # DiskTensorHandle. Never concatenated (a cat would transiently double the tens-of-GB
        # footprint); lookups route rows to shards instead. All shards but the last hold
        # rows_per_shard rows
        self.tables = None
        self.handles = None
        self.rows_per_shard = None
        self.num_rows = 0
        self.head_bias = None
        self.head_offsets = None
        self.head_vocab_sizes = None
        self.layer_multipliers = None
        self.codebook = None
        self._pins = []             # pinned staging sets for the fast path (see _PinSet)
        self._pending = []          # queued prefetches, oldest first: {"history", "pin", "future"}
        self._executor = None
        self.prefetch_stats = {"hit": 0, "miss": 0, "retired": 0}
        self._prof = {"calls": 0, "rows": 0, "hash": 0.0, "gather": 0.0} if PROF_ENABLED else None
        self._cache = None          # row cache in front of the disk path (see _cache_mib)
        self._cache_share_mib = 0   # this instance's part of the process-wide budget
        self._cache_stats = {"hit": 0, "miss": 0}
        self._cache_lock = threading.Lock()
        self._row_dtype = None      # stored row dtype of the unquantized table

        self.caps.update({"prefer_cpu": True})

    @override
    def optimizer_targets(self):
        return []

    def _load_aux(self, names: dict[str, str]):
        stc = self.config.stc
        def get(name, optional = False):
            # no_defer: these are consumed (copied) during load(), before a deferred load pass
            # would have filled them
            t = stc.get_tensor(name, "cpu", optional = True, allow_bf16 = True, no_defer = True)
            if t is None and not optional:
                raise ValueError(f"Required tensor {name} not found for {self.key}")
            return t
        # The hashing runs on the CPU (where the token ids live), so the hash parameters stay
        # host-side; only the dequant bias goes to the device
        self.head_offsets = get(names["offsets"]).long().contiguous()
        self.head_vocab_sizes = get(names["sizes"]).long().contiguous()
        self.layer_multipliers = get(names["multipliers"]).long().contiguous()
        bias = get(names["bias"], optional = True) if "bias" in names else None
        self.head_bias = bias.half().contiguous().to(self.device) if bias is not None else None
        assert self.head_offsets.shape[0] == self.num_heads
        # These were buffered reads of the table file. On Windows a buffered file object (even a
        # closed one, for a few seconds) throttles the unbuffered row gathers on the same file, so
        # release the loader's handle now rather than at the end of the load
        for f in {stc.tensor_file_map[n] for n in names.values() if n in stc.tensor_file_map}:
            stc.release_file(f)

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        stc = self.config.stc
        parent = self.key.rsplit(".", 1)[0]

        def enumerate_shards(suffix):
            keys = []
            while stc.has_tensor(f"{self.key}.shard_{len(keys)}.{suffix}"):
                keys.append(f"{self.key}.shard_{len(keys)}.{suffix}")
            return keys

        def shard_shapes(keys):
            # all shards but the last must hold the same row count (row -> shard routing is a
            # plain division); the last may be short
            shapes = [stc.get_tensor_meta(k)[k]["shape"] for k in keys]
            assert all(s[0] == shapes[0][0] for s in shapes[:-1]), \
                "n-gram table shards must have equal row counts (last may be short)"
            assert shapes[-1][0] <= shapes[0][0]
            self.rows_per_shard = shapes[0][0]
            self.num_rows = sum(s[0] for s in shapes)
            return shapes

        trellis_keys = enumerate_shards("trellis")
        if not trellis_keys and stc.has_tensor(f"{self.key}.trellis"):
            trellis_keys = [f"{self.key}.trellis"]    # single-tensor layout of older files

        if trellis_keys:
            # quantized table
            shapes = shard_shapes(trellis_keys)
            words = shapes[0][1]
            self.K = (words - 1) * 16 // ROW_DIM
            assert words == words_per_row(self.K)
            assert all(s[1] == words for s in shapes)
            self._load_aux({
                "offsets": f"{self.key}.head_offsets",
                "sizes": f"{self.key}.head_vocab_sizes",
                "multipliers": f"{self.key}.layer_multipliers",
                "bias": f"{self.key}.head_bias",
            })
            self.codebook = mul1_codebook(device)
            keys = trellis_keys

        else:
            # unquantized source table: single tensor or shard_N split
            if stc.has_tensor(f"{self.key}.weight"):
                keys = [f"{self.key}.weight"]
            else:
                keys = enumerate_shards("weight")
                if not keys:
                    raise ValueError(f"No .trellis, .weight or .shard_N.weight tensors found for {self.key}")
            self._load_aux({
                "offsets": f"{parent}.ngram_heads_offsets",
                "sizes": f"{parent}.ngram_heads_vocab_sizes",
                "multipliers": f"{parent}.layer_multipliers",
            })
            shapes = shard_shapes(keys)
            assert all(s[1] == ROW_DIM for s in shapes)

        quantized = trellis_keys != []
        stream_from_disk = self.stream_from_disk
        if stream_from_disk is None:
            infer_params = getattr(self.config, "infer_params", None)
            stream_from_disk = infer_params.ngram_stream_from_disk if infer_params is not None else None
        if stream_from_disk is None:
            # Auto: per-token row preads cost ~0.8 ms of GPU idle per decode token (tokplan1), so hold
            # the table in RAM when it fits with a wide margin; otherwise stream from disk
            from ..util.memory import host_memory_available
            avail = host_memory_available()
            tbl = self.num_rows * (words_per_row(self.K) * 2 if quantized else ROW_DIM * 2)
            stream_from_disk = avail is None or avail < tbl + (48 << 30)
        if stream_from_disk:
            self.mode = "trellis_disk" if quantized else "fp16_disk"
            self.handles = [stc.get_tensor_handle(k) for k in keys]
            if not quantized:
                self._row_dtype = self.handles[0].dtype
            # Shards that sit back-to-back in one file (the layout convert_ngram.py writes)
            # collapse into a single handle spanning the whole table: _gather_rows issues one
            # synchronous gather call per handle segment
            h0 = self.handles[0]
            if len(self.handles) > 1 and all(
                h.filename == h0.filename and h.row_bytes == h0.row_bytes and
                h.abs_offset == h0.abs_offset + s * self.rows_per_shard * h0.row_bytes
                for s, h in enumerate(self.handles)
            ):
                merged = DiskTensorHandle(
                    key = self.key, filename = h0.filename, abs_offset = h0.abs_offset,
                    shape = [self.num_rows, *h0.row_shape], dtype = h0.dtype)
                stc.find_stc(keys[0]).disk_handles.append(merged)   # closed with the collection
                self.handles = [merged]
                self.rows_per_shard = self.num_rows
            if os.name == "nt":
                # Release the loader's handles to the table files now
                for h in set(h.filename for h in self.handles):
                    stc.release_file(h)
        else:
            self.mode = "trellis_ram" if quantized else "fp16_ram"
            if len(keys) == 1:
                self.tables = [stc.get_tensor(keys[0], "cpu", allow_bf16 = not quantized, no_defer = True)]
            else:
                # Sharded table: one contiguous slab, each shard copied into its slice as it loads
                slab = None
                for s_i, k in enumerate(keys):
                    t = stc.get_tensor(k, "cpu", allow_bf16 = not quantized, no_defer = True)
                    if slab is None:
                        from ..util.memory import check_host_memory
                        check_host_memory(self.num_rows * t[0].numel() * t.element_size(),
                                          f"n-gram table {self.key} held in RAM (--ngram_ram)")
                        slab = torch.empty((self.num_rows, *t.shape[1:]), dtype = t.dtype)
                    r0 = s_i * self.rows_per_shard
                    slab[r0 : r0 + t.shape[0]].copy_(t)
                    del t
                self.tables = [slab]
                self.rows_per_shard = self.num_rows
            if not quantized:
                self._row_dtype = self.tables[0].dtype
        if not getattr(NGramEmbedding, "_mode_logged", False):
            NGramEmbedding._mode_logged = True
            from ..util.memory import host_memory_available
            av = host_memory_available()
            print(f"[ngram] table mode {self.mode}: {self.num_rows} rows, host MemAvailable "
                  f"{'?' if av is None else f'{av / 2**30:.1}'} GiB at load (RAM mode needs table + 48 GiB; "
                  f"disk mode = every prefill chunk reads its rows from the file, a busy disk or a cold page cache stalls it)",
                  flush=True)

    def ensure_resident(self, min_swap_bytes: int = 64 << 20) -> dict:
        """RAM modes only. The table is anonymous host memory that the kernel may have swapped out during the load (the weights
        take most of the RAM): every later row lookup that lands on a swapped page then waits for a disk read. If this process has
        more than min_swap_bytes in swap, read the whole table back in (MADV_POPULATE_READ). Contents are unchanged"""
        info = {"mode": self.mode, "swap_before": process_swap_bytes(), "populated": 0, "seconds": 0.0}
        if self.tables is None or info["swap_before"] is None or info["swap_before"] < min_swap_bytes:
            return info
        import time
        t0 = time.perf_counter()
        for t in self.tables:
            n = _madvise_range(t, _MADV_POPULATE_READ)
            if n < 0:
                info["error"] = "madvise(MADV_POPULATE_READ) failed"
                break
            info["populated"] += n
        info["seconds"] = round(time.perf_counter() - t0, 2)
        info["swap_after"] = process_swap_bytes()
        return info

    @override
    def unload(self):
        self._drain_prefetch()      # queued workers still read the table; first
        self.device = None
        self.mode = None
        self.tables = None
        self.handles = None
        self._pins = []
        self._cache_release()
        self._row_dtype = None
        self.head_bias = None
        self.head_offsets = None
        self.head_vocab_sizes = None
        self.layer_multipliers = None
        self.codebook = None

    @override
    def get_tensors(self):
        # The table is never resident as a whole in the general case; export/compile of this
        # module is handled at conversion time, not here
        return {}

    @override
    def weights_numel(self):
        return self.num_rows * ROW_DIM

    def _fetch_packed(self, uids_cpu: torch.Tensor) -> torch.Tensor:
        """Gather rows of the backing store (packed int16 or raw fp16/bf16) to CPU, routing
        global row indices to the individual shard tensors/handles."""
        ram = self.tables is not None
        store = self.tables if ram else self.handles

        def gather(s, local):
            return store[s].index_select(0, local) if ram else store[s].read_rows(local)

        if len(store) == 1:
            return gather(0, uids_cpu)
        shard = uids_cpu // self.rows_per_shard
        local = uids_cpu - shard * self.rows_per_shard
        out = None
        for s in shard.unique().tolist():
            m = shard == s
            rows = gather(s, local[m])
            if out is None:
                out = torch.empty((uids_cpu.numel(), *rows.shape[1:]), dtype = rows.dtype)
            out[m] = rows
        return out

    def fetch_rows(self, uids: torch.Tensor, out_dtype: torch.dtype = torch.half) -> torch.Tensor:
        """Unique row indices (any device) -> decoded (N, 160) rows on the module's device.
        Reference form of the row pipeline (torch codec); forward() runs the fast path."""
        uids_cpu = uids.to("cpu", torch.int64)
        raw = self._fetch_packed(uids_cpu).to(self.device)
        if self.mode.startswith("trellis"):
            heads = (torch.searchsorted(self.head_offsets.to(self.device),
                                        uids.to(self.device, torch.int64),
                                        right = True) - 1).clamp(0, self.num_heads - 1)
            rows = dequant_rows(raw, self.K, self.codebook, self.head_bias.float()[heads])
        else:
            rows = raw.float()
        return rows.to(out_dtype)

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        # mirrors the reference implementation: n-grams never span an eos boundary; positions
        # whose shifted source would cross one read eos instead
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device = token_ids.device)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim = 1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim = 1)
        position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim = 1, index = gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def compute_ngram_ids(self, token_history: torch.Tensor, out_len: int) -> torch.Tensor:
        """
        token_history: (bsz, context + seq_len) token ids including the (ngram_size - 1) tokens
        preceding the sequence (eos-padded at the start of a new sequence).
        Returns (bsz, out_len, num_heads) global table row indices for the last out_len positions.
        """
        th = token_history.long()
        shifted = [self._shift_right_ignore_eos(th, s) for s in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            lo = (ngram - 2) * self.heads_per_ngram
            hi = lo + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0].to(th.device)
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[position] * self.layer_multipliers[position].to(th.device))
            sizes = self.head_vocab_sizes[lo:hi].to(th.device)
            offsets = self.head_offsets[lo:hi].to(th.device)
            blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes.view(1, 1, -1)) + offsets.view(1, 1, -1))
        return torch.cat(blocks, dim = -1)[:, -out_len:]

    def embed_ids(self, ngram_ids: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """(bsz, seq_len, num_heads) row indices -> (bsz, seq_len, ple_embed_dim) on device.
        Reference form; forward() runs the fast path."""
        bsz, seq_len, H = ngram_ids.shape
        flat = ngram_ids.reshape(-1)
        uids, inverse = torch.unique(flat, return_inverse = True)
        rows = self.fetch_rows(uids, out_dtype or self.out_dtype or torch.half)
        out = rows[inverse.to(self.device)]
        return out.view(bsz, seq_len, H * ROW_DIM)

    def forward_reference(self, x: torch.Tensor, params: dict,
                          out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Pure-torch reference pipeline (hashing + codec), kept for tests and A/B."""
        out_len = x.shape[1] - self.context_len
        ngram_ids = self.compute_ngram_ids(x, out_len)
        return self.embed_ids(ngram_ids, out_dtype)

    # ---- fast path -----------------------------------------------------------------------------
    #
    # Hashing, eos segmentation and dedup run in one C++ call on the CPU, where the token ids
    # already live (pinned in the generator, so nothing round-trips through the device); the
    # unique rows are gathered with threaded preads (disk modes) or index_select (RAM modes)
    # into a pinned staging set; one non-blocking H2D then feeds the GPU trellis dequant kernel
    # (or a plain upcast for unquantized tables) and the inverse gather.
    #
    # The staging step depends only on the token ids, so it can run ahead of the forward on a
    # worker thread (prefetch()): the model stages the chunk before its first layers are issued,
    # and the cold gather (~90 ms per 4096-token chunk, vs ~5 ms page-cache-warm) overlaps block
    # 0 instead of stalling at this layer. Each queued prefetch owns one staging set; forward()
    # takes the set whose staged history equals the one it was given and stages inline
    # otherwise, so a prefetch for the wrong ids can only cost time. A set's CUDA event marks
    # the last forward's uploads from it; the next writer waits on it before reusing the buffers.
    # Decode-sized inputs stay inline: their 16-row gathers are already parallel preads, and the
    # thread hop per token measured as a net loss.

    def _drain_prefetch(self):
        for e in list(self._pending):
            self._retire(e)
        if self._executor is not None:
            self._executor.shutdown(wait = True)
            self._executor = None

    def _forget(self, entry: dict):
        # by identity: list.remove would compare the entries' history tensors
        self._pending = [e for e in self._pending if e is not entry]

    def _retire(self, entry: dict):
        # Drop a queued prefetch that no forward will take. Its worker may still be writing the
        # staging set, so wait it out (a cold gather, at most) unless it hasn't started
        self._forget(entry)
        self.prefetch_stats["retired"] += 1
        f = entry["future"]
        if not f.cancel():
            f.result()
        entry["pin"].held = False

    def _acquire_pin(self, n: int, row_words: int, row_dtype: torch.dtype) -> _PinSet:
        """A staging set not held by a queued prefetch, grown to n ids."""
        free = [p for p in self._pins if not p.held]
        if not free and len(self._pins) < MAX_PIN_SETS:
            free = [_PinSet()]
            self._pins += free
        if not free:
            # every set is held by a queued prefetch: retire one, preferably one whose staging
            # already finished (a stale guess), else the oldest
            done = [e for e in self._pending if e["future"].done()]
            self._retire(done[0] if done else self._pending[0])
            free = [p for p in self._pins if not p.held]
        pin = free[0]
        pin.grow(n, row_words, row_dtype)
        pin.held = True
        return pin

    @torch.inference_mode()
    def _stage(self, history: torch.Tensor, pin: _PinSet) -> int:
        """Hash + gather for a (bsz, context + seq) id history into pin; returns the unique row
        count. Runs on the prefetch worker or inline (inference mode is thread-local, and the
        staging buffers are inference tensors)."""
        out_len = history.shape[1] - self.context_len
        if pin.event is not None:
            # the previous forward's non_blocking uploads read this set; the generator issues
            # chunk forwards back to back with no host sync, so wait before rewriting it
            pin.event.synchronize()
        if self._prof is None:
            U = ext.ngram_hash_cpu(
                history, out_len, self.layer_multipliers, self.head_offsets, self.head_vocab_sizes,
                self.heads_per_ngram, self.eos_token_id,
                pin.uids, pin.inverse, pin.heads)
            self._gather_rows(pin.uids[:U], pin.packed[:U])
            return U
        import time
        t0 = time.perf_counter()
        U = ext.ngram_hash_cpu(
            history, out_len, self.layer_multipliers, self.head_offsets, self.head_vocab_sizes,
            self.heads_per_ngram, self.eos_token_id,
            pin.uids, pin.inverse, pin.heads)
        t1 = time.perf_counter()
        self._gather_rows(pin.uids[:U], pin.packed[:U])
        t2 = time.perf_counter()
        p = self._prof
        p["calls"] += 1
        p["rows"] += U
        p["hash"] += t1 - t0
        p["gather"] += t2 - t1
        if p["calls"] % 64 == 0:
            c = p["calls"]
            ch, cm = self._cache_stats["hit"], self._cache_stats["miss"]
            hits = f", cache {100 * ch / (ch + cm):.0}% of {ch + cm} rows" if ch + cm else ""
            print(f"[ngram prof] {self.mode}: {c} stages, {p['rows'] / c:.1} rows/stage, "
                  f"hash {1000 * p['hash'] / c:.3} ms, gather {1000 * p['gather'] / c:.3} ms"
                  f" ({'disk' if self.tables is None else 'ram'}{hits})", flush=True)
        return U

    def _match(self, history: torch.Tensor) -> dict | None:
        for e in self._pending:
            if e["history"].shape == history.shape and torch.equal(e["history"], history):
                return e
        return None

    def _cache_rows_init(self, row_words: int, row_dtype: torch.dtype):
        """Allocate the host row cache on first use (n_slots x row_words, FIFO eviction), taking this
        instance's share of the process-wide budget. Two staging sets can start at the same time, so
        only the first one to take the lock allocates."""
        if CACHE_MIB <= 0:
            return
        with self._cache_lock:
            if self._cache is not None:
                return
            with _cache_budget_lock:
                mib = min(CACHE_MIB, _cache_budget_mib[0])
                if mib <= 0:
                    print("[ngram] row cache budget is used up by the earlier PLE layers; this layer "
                          "reads the disk path uncached (EXL3_NGRAM_CACHE_MB sets the process total)",
                          flush = True)
                    return
                _cache_budget_mib[0] -= mib
            slot_bytes = row_words * torch.empty(0, dtype = row_dtype).element_size()
            slots = max(1, min((mib << 20) // max(1, slot_bytes), self.num_rows))
            nbytes = slots * slot_bytes
            self._cache = {
                "rows": torch.empty((slots, row_words), dtype = row_dtype),
                "uids": [-1] * slots,
                "index": {},
                "ring": 0,
                "slots": slots,
                "bytes": nbytes,
            }
            # charged in MiB, rounded up: a slot count capped by num_rows leaves part of the share
            # unused, and the next layer should get it
            used = -(-nbytes >> 20)
            with _cache_budget_lock:
                _cache_budget_mib[0] += mib - used
            self._cache_share_mib = used
        print(f"[ngram] row cache {nbytes / 2**20:.0} MiB ({slots} rows) in front of the disk path, "
              f"{CACHE_MIB - _cache_budget_mib[0]} of {CACHE_MIB} MiB of the process-wide budget in use "
              f"(EXL3_NGRAM_CACHE_MB=0 disables)", flush = True)

    def _cache_release(self):
        """Drop the row cache and hand this instance's share of the budget back (unload(), so a later
        load() of another layer can use it)."""
        if self._cache_share_mib:
            with _cache_budget_lock:
                _cache_budget_mib[0] += self._cache_share_mib
            self._cache_share_mib = 0
        self._cache = None

    def _cache_plan(self, uids: torch.Tensor, out: torch.Tensor):
        """Split the (sorted) uids into cache hits and misses, copying the hit rows into `out`
        while the lock is held: a fill publishes a slot only together with its data, and an
        eviction cannot take a slot between the index lookup and the copy. In a tiny input list,
        so no vectorized form is needed; this costs ~10 us against the ~40 us of SSD latency per
        miss. Returns (miss_pos, miss_uids)."""
        idx = self._cache["index"]
        hit_pos, hit_slots, miss_pos, miss_uids = [], [], [], []
        with self._cache_lock:
            for i, u in enumerate(uids.tolist()):
                s = idx.get(u)
                if s is None:
                    miss_pos.append(i)
                    miss_uids.append(u)
                else:
                    hit_pos.append(i)
                    hit_slots.append(s)
            if hit_pos:
                out.index_copy_(0, torch.tensor(hit_pos, dtype = torch.int64),
                                self._cache["rows"].index_select(0, torch.tensor(hit_slots, dtype = torch.int64)))
            self._cache_stats["hit"] += len(hit_pos)
        return (torch.tensor(miss_pos, dtype = torch.int64),
                torch.tensor(miss_uids, dtype = torch.int64))

    def _cache_fill(self, miss_uids: torch.Tensor, rows: torch.Tensor):
        """Insert gathered rows, evicting FIFO. A concurrent stage may have filled some of them
        (two staging sets can be in flight); those are skipped rather than duplicated. The slot is
        published in the index and written under one lock, so no reader can plan a hit on it
        before its data is there (and an eviction cannot hand a reader a recycled slot)."""
        c = self._cache
        idx, cuids, n = c["index"], c["uids"], c["slots"]
        uids = miss_uids.tolist()
        want_slots, want_rows = [], []
        with self._cache_lock:
            for i, u in enumerate(uids):
                if u in idx:
                    continue
                s = c["ring"]
                c["ring"] = s + 1 if s + 1 < n else 0
                old = cuids[s]
                if old >= 0:
                    idx.pop(old, None)
                cuids[s] = u
                idx[u] = s
                want_slots.append(s)
                want_rows.append(i)
            if want_slots:
                c["rows"].index_copy_(0, torch.tensor(want_slots, dtype = torch.int64),
                                      rows.index_select(0, torch.tensor(want_rows, dtype = torch.int64)))
        self._cache_stats["miss"] += len(uids)

    def _gather_rows(self, uids: torch.Tensor, out: torch.Tensor):
        """Gather the (sorted) unique rows into the pinned staging buffer. Disk modes go through
        the host row cache; RAM modes never need it."""
        if self.tables is None and CACHE_MIB > 0 and uids.numel() <= CACHE_MAX_ROWS:
            if self._cache is None:
                self._cache_rows_init(out.shape[1], out.dtype)
            if self._cache is not None:
                miss_pos, miss_uids = self._cache_plan(uids, out)
                if miss_uids.numel():
                    rows = torch.empty((miss_uids.numel(), out.shape[1]), dtype = out.dtype)
                    self._gather_store(miss_uids, rows)
                    out.index_copy_(0, miss_pos, rows)
                    self._cache_fill(miss_uids, rows)
                return
        self._gather_store(uids, out)

    def _gather_store(self, uids: torch.Tensor, out: torch.Tensor):
        """Gather the (sorted) unique rows from the backing store, routing shard segments
        (contiguous in the sorted list) to their tensor/handle."""
        stores = self.tables if self.tables is not None else self.handles
        if len(stores) > 1:
            # One searchsorted for every shard boundary (a per-shard call costs ~2.5 us each)
            bounds = torch.arange(1, len(stores), dtype = torch.int64) * self.rows_per_shard
            cuts = torch.searchsorted(uids, bounds).tolist() + [uids.numel()]
        else:
            cuts = [uids.numel()]
        i0 = 0
        for s, store in enumerate(stores):
            i1 = cuts[s]
            if i1 > i0:
                seg = uids[i0 : i1]
                base = s * self.rows_per_shard
                if self.tables is not None:
                    torch.index_select(store, 0, seg - base if base else seg,
                                       out = out[i0 : i1])
                else:
                    gather = getattr(ext, "ngram_gather_cpu", None)
                    if gather is not None:
                        gather(store._ensure_open(), store.abs_offset,
                               store.row_bytes, seg.contiguous(), base, out[i0 : i1])
                    else:
                        out[i0 : i1].copy_(store.read_rows(seg - base if base else seg))
            i0 = i1

    def prefetch(self, history: torch.Tensor):
        """
        Stage the rows for a coming forward over `history`, the exact (bsz, context + seq) id
        history that forward() will receive, on a worker thread. Decode-sized inputs stage inline
        faster than the thread hop, so those are ignored; a history already queued is ignored too.
        """
        if not PREFETCH_ENABLED or self.mode is None or history.dim() != 2:
            return
        bsz, out_len = history.shape[0], history.shape[1] - self.context_len
        if out_len <= 0 or bsz * out_len < PREFETCH_MIN_TOKENS:
            return
        history = history.to("cpu", torch.int64).contiguous().clone()
        if self._match(history) is not None:
            return
        trellis = self.mode.startswith("trellis")
        pin = self._acquire_pin(
            bsz * out_len * self.num_heads,
            words_per_row(self.K) if trellis else ROW_DIM,
            torch.int16 if trellis else self._row_dtype)
        for h in self.handles or []:
            h._ensure_open()        # lazy open isn't thread-safe; do it here, not on the worker
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers = 1, thread_name_prefix = "ngram_prefetch")
        self._pending.append({
            "history": history,
            "pin": pin,
            "future": self._executor.submit(self._stage, history, pin),
        })

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """
        x: (bsz, context + seq_len) token history (CPU in the hot path); returns embeddings for
        the last seq_len = x.shape[1] - context_len positions, on the module's device.
        """
        out_len = x.shape[1] - self.context_len
        trellis = self.mode.startswith("trellis")
        if not hasattr(ext, "ngram_hash_cpu") or \
                (trellis and not hasattr(ext, "ngram_dequant")):
            return self.forward_reference(x, params, out_dtype)

        ids = x.to("cpu", torch.int64).contiguous()
        bsz = ids.shape[0]
        H = self.num_heads
        n = bsz * out_len * H
        dev = self.device
        row_words = words_per_row(self.K) if trellis else ROW_DIM
        row_dtype = torch.int16 if trellis else self._row_dtype

        entry = self._match(ids)
        if entry is not None:
            self._forget(entry)
            self.prefetch_stats["hit"] += 1
            pin = entry["pin"]
            U = entry["future"].result()
        else:
            self.prefetch_stats["miss"] += 1
            pin = self._acquire_pin(n, row_words, row_dtype)
            U = self._stage(ids, pin)

        packed_d = pin.packed[:U].to(dev, non_blocking = True)
        inv_d = pin.inverse[:n].to(dev, non_blocking = True)
        if trellis:
            heads_d = pin.heads[:U].to(dev, non_blocking = True)
            rows = torch.empty((U, ROW_DIM), dtype = torch.half, device = dev)
            ext.ngram_dequant(packed_d, self.K, heads_d, self.head_bias, rows)
        else:
            rows = packed_d.float()
        if rows.is_cuda:
            pin.event = torch.cuda.Event()
            pin.event.record(torch.cuda.current_stream(rows.device))
        pin.held = False
        out = rows.index_select(0, inv_d).view(bsz, out_len, H * ROW_DIM)
        dt = out_dtype or self.out_dtype or torch.half
        return out.to(dt)
