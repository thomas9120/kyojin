"""The host row cache in front of the n-gram disk path (NGramEmbedding._gather_rows).

CPU-only: the backing store is faked, so this tests hit/miss routing, eviction and the bypass for
prefill-sized inputs rather than any I/O. The last two tests cover the two-thread case the module
documents (prefetch worker + inline stage): a slot must never be readable before its data is there.
"""
import random
import threading
import time
from typing import Any, cast

import torch

from exllamav3.modules import ngram_embedding as ne

ROW_WORDS = 2


def _disk_module(rows: int) -> Any:
    # a stub, not a constructed module: untyped on purpose (only the cache fields exist)
    m = cast(Any, object.__new__(ne.NGramEmbedding))
    m.tables = None                 # disk path -> the cache applies
    m.handles = ["fake"]
    m.rows_per_shard = rows
    m.num_rows = rows
    m.K = 1
    m._cache = None
    m._cache_stats = {"hit": 0, "miss": 0}
    m._cache_lock = threading.Lock()
    m.stored = []

    def fake_store(uids, out):
        m.stored.append(uids.tolist())
        for i, u in enumerate(uids.tolist()):
            out[i] = u              # row content = its uid, so assembly errors are visible
    m._gather_store = fake_store
    return m


def _gather(m: Any, uids):
    out = torch.zeros((len(uids), ROW_WORDS), dtype = torch.int16)
    m._gather_rows(torch.tensor(uids, dtype = torch.int64), out)
    return out[:, 0].tolist()


def test_second_read_of_a_row_does_not_touch_the_store(monkeypatch):
    monkeypatch.setattr(ne, "CACHE_MIB", 1)
    m = _disk_module(rows = 1 << 20)
    assert _gather(m, [3, 5, 7]) == [3, 5, 7]
    assert m.stored == [[3, 5, 7]]
    assert _gather(m, [5, 7, 9]) == [5, 7, 9]          # 5 and 7 cached, 9 new
    assert m.stored == [[3, 5, 7], [9]]
    assert m._cache_stats == {"hit": 2, "miss": 4}


def test_eviction_keeps_the_most_recent_rows(monkeypatch):
    # 2 slots only: 4 rows * 2 int16 = 16 B -> 1 MiB / 16 B is capped by num_rows, so size the
    # cache through the row count instead: rows = 2
    monkeypatch.setattr(ne, "CACHE_MIB", 1)
    m = _disk_module(rows = 2)                         # slots = min(huge, num_rows) = 2
    assert _gather(m, [0, 1]) == [0, 1]
    assert _gather(m, [2, 3]) == [2, 3]                # evicts 0 and 1
    assert _gather(m, [2, 3]) == [2, 3]                # both hits, no store call
    assert m.stored == [[0, 1], [2, 3]]
    assert _gather(m, [0]) == [0]                      # evicted again -> re-read
    assert m.stored == [[0, 1], [2, 3], [0]]


def test_prefill_sized_inputs_bypass_the_cache(monkeypatch):
    monkeypatch.setattr(ne, "CACHE_MIB", 1)
    m = _disk_module(rows = 1 << 20)
    uids = list(range(ne.CACHE_MAX_ROWS + 1))
    assert _gather(m, uids) == uids
    assert m._cache is None                            # never allocated
    assert m._cache_stats == {"hit": 0, "miss": 0}


class _SlowRows:
    """The cache's row tensor (`rows`) with a fill that parks between the index update and the row
    write, so a second thread is guaranteed to plan while a slot is half-published (the pre-fix
    race). Only the two methods _cache_plan / _cache_fill use are forwarded."""

    def __init__(self, rows, started: threading.Event, release: threading.Event):
        self.rows, self.started, self.release = rows, started, release

    def index_select(self, dim, index):
        return self.rows.index_select(dim, index)

    def index_copy_(self, dim, index, source):
        self.started.set()
        self.release.wait(5.0)
        return self.rows.index_copy_(dim, index, source)


def test_a_published_slot_is_never_read_before_its_data(monkeypatch):
    # thread A fills 7 and parks before writing the row; thread B must not get the stale slot
    monkeypatch.setattr(ne, "CACHE_MIB", 1)
    m = _disk_module(rows = 1 << 20)
    started, release = threading.Event(), threading.Event()
    results = {}

    def fill():
        results["a"] = _gather(m, [7])

    def hit():
        results["b"] = _gather(m, [7])

    m._cache_rows_init(ROW_WORDS, torch.int16)
    m._cache["rows"] = _SlowRows(m._cache["rows"], started, release)
    a = threading.Thread(target = fill)
    a.start()
    assert started.wait(5.0), "fill never reached the row write"
    b = threading.Thread(target = hit)
    b.start()
    time.sleep(0.2)                     # B has planned and copied by now (or is blocked, post-fix)
    release.set()
    a.join(5.0)
    b.join(5.0)
    assert results["a"] == [7]
    assert results["b"] == [7], "a slot was handed out before its row was written"
    assert m.stored == [[7]]            # B hit the slot, it did not go back to the store


def test_concurrent_staging_keeps_rows_matching_their_uids(monkeypatch):
    # two threads over one small cache: eviction and fill of the same uids are the documented
    # concurrent case; every gathered row must still be the row of its uid
    monkeypatch.setattr(ne, "CACHE_MIB", 1)
    m = _disk_module(rows = 64)         # 64 slots for 256 uids: constant eviction
    barrier = threading.Barrier(2)
    errors = []

    def worker(seed):
        rng = random.Random(seed)
        barrier.wait(5.0)
        for _ in range(200):
            uids = sorted(rng.sample(range(256), rng.randint(1, 24)))
            got = _gather(m, uids)
            if got != uids:
                errors.append((uids, got))

    ts = [threading.Thread(target = worker, args = (s,)) for s in (1, 2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30.0)
    assert not errors, f"row cache served a foreign row: {errors[:3]}"


def test_cache_off_by_default_outside_windows(monkeypatch):
    monkeypatch.delenv("EXL3_NGRAM_CACHE_MB", raising = False)
    monkeypatch.setattr(ne.os, "name", "posix")
    assert ne._cache_mib() == 0
    monkeypatch.setattr(ne.os, "name", "nt")
    assert ne._cache_mib() == 1024
    monkeypatch.setenv("EXL3_NGRAM_CACHE_MB", "0")
    assert ne._cache_mib() == 0
    monkeypatch.setenv("EXL3_NGRAM_CACHE_MB", "64")
    assert ne._cache_mib() == 64
