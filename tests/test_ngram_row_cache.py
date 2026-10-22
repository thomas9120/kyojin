"""The host row cache in front of the n-gram disk path (NGramEmbedding._gather_rows).

CPU-only: the backing store is faked, so this tests hit/miss routing, eviction and the bypass for
prefill-sized inputs rather than any I/O.
"""
import threading

import torch

from exllamav3.modules import ngram_embedding as ne

ROW_WORDS = 2


def _disk_module(rows: int):
    m = object.__new__(ne.NGramEmbedding)
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


def _gather(m, uids):
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
