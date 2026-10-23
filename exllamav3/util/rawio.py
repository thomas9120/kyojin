"""Positioned ("pread") reads of raw rows for the disk-backed tiers (expert store, Engram readers).

One primitive, two implementations: POSIX uses os.preadv on a plain fd, Windows uses the overlapped
Win32 HANDLEs that loader/safetensors.py maintains for DiskTensorHandle (the OVERLAPPED offset is
the pread equivalent and a handle can be read from several threads at once).
os.O_DIRECT and FILE_FLAG_NO_BUFFERING are the same idea on the two platforms: bypass the page
cache. Whether the caller's offsets and buffers must be sector-aligned follows from that flag; this
module copies and aligns nothing.

The module exists because os.O_DIRECT and os.preadv do not exist on Windows: naming them
unconditionally raises AttributeError at *attribute access*, before any I/O is attempted, which no
`except OSError` around the read can catch. Here the platform is resolved once, and a platform
without unbuffered reads reports it through a normal OSError at open time.
"""
from __future__ import annotations
import os
from typing import Any

NT = os.name == "nt"
_O_DIRECT = getattr(os, "O_DIRECT", 0)

if NT:
    # imported per call, not at module scope: the loader imports the compiled extension, and the
    # tiers that use this module (and their CPU tests) must stay importable without it
    def _open(path: str, direct: bool) -> tuple[int, bool]:
        from ..loader.safetensors import _win_open_ref, _win_open_stream
        if direct:
            return _win_open_stream(path), True
        return _win_open_ref(path), False

    def pread_into(handle: int, mv: memoryview, offset: int) -> int:
        """Positioned read into `mv`; may return short, like pread. Thread-safe on a shared
        handle (the offset travels in the OVERLAPPED, not in handle state)."""
        from ..loader.safetensors import _win_pread_into
        return _win_pread_into(handle, mv, offset)

    def close_rows(handle: int):
        from ..loader.safetensors import _k32
        _k32.CloseHandle(handle)

else:
    _preadv: Any = getattr(os, "preadv")

    def _open(path: str, direct: bool) -> tuple[int, bool]:
        if direct:
            return os.open(path, os.O_RDONLY | _O_DIRECT), True
        return os.open(path, os.O_RDONLY), False

    def pread_into(handle: int, mv: memoryview, offset: int) -> int:
        """Positioned read into `mv`; may return short, like pread."""
        return _preadv(handle, [mv], offset)

    def close_rows(handle: int):
        os.close(handle)


def direct_supported() -> bool:
    """True when this platform can bypass the page cache for positioned reads."""
    return NT or _O_DIRECT != 0


def open_rows(path: str, direct: bool = True, allow_buffered: bool = False) -> tuple[int, bool]:
    """Open `path` for positioned reads. Returns (handle, direct), where `direct` is True only when
    the unbuffered path was actually taken. Falls back to buffered reads when `allow_buffered`,
    otherwise raises OSError (platform or filesystem without the flag)."""
    if direct and not direct_supported():
        if not allow_buffered:
            raise OSError(f"{path}: unbuffered positioned reads are not available on this platform")
        direct = False
    if not direct:
        return _open(path, False)
    try:
        return _open(path, True)
    except OSError:
        if not allow_buffered:
            raise
    return _open(path, False)


def drop_cache(handle: int):
    """Ask the kernel to forget the pages a buffered reader touched; no-op where there is no
    posix_fadvise (buffered reads on Windows keep their pages)."""
    advise = getattr(os, "posix_fadvise", None)
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or advice is None:
        return
    try:
        advise(handle, 0, 0, advice)
    except OSError:
        pass
