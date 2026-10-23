"""Unit tests for exllamav3.util.hip_lib (finding the HIP runtime next to torch).

The directory-scan half is platform independent, so it runs everywhere against NAME, the library name
this platform's wheel uses. The `/proc/self/maps` fallback is Linux-only and the Windows branch
searches the ROCm wheel's DLL trees instead, so each of those gets skipped on the other platform
rather than asserting POSIX behaviour on Windows.
"""

import importlib.util, io, os, contextlib, sys, types
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "hip_lib", Path(__file__).parent.parent / "exllamav3" / "util" / "hip_lib.py")
assert _spec is not None and _spec.loader is not None      # the module ships with this repo
hip_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hip_lib)

NAME = hip_lib._NAME            # libamdhip64.so (POSIX) or amdhip64 (Windows)


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"")


def _find(d):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        return hip_lib.find_hip_runtime(str(d)), err.getvalue()


def test_plain_name_wins_and_is_silent(tmp_path):
    _touch(tmp_path, NAME, NAME + ".7")
    path, log = _find(tmp_path)
    assert path == str(tmp_path / NAME) and log == ""


def test_versioned_fallback_picks_highest_and_logs_once(tmp_path):
    _touch(tmp_path, NAME + ".6", NAME + ".7", NAME + ".7.1.2", NAME + ".10",
           NAME + ".bak", NAME + ".7.txt")
    path, log = _find(tmp_path)
    assert path == str(tmp_path / (NAME + ".10"))
    assert log.count("\n") == 1 and (NAME + ".10") in log
    assert _find(tmp_path)[1] == ""   # already reported


def test_dotted_versions_sort_numerically(tmp_path):
    _touch(tmp_path, NAME + ".7", NAME + ".7.1.2", NAME + ".7.1.10")
    assert _find(tmp_path)[0] == str(tmp_path / (NAME + ".7.1.10"))


def test_none_when_absent_and_load_raises(tmp_path):
    assert _find(tmp_path)[0] is None
    with pytest.raises(OSError):
        hip_lib.load_hip_runtime(str(tmp_path))


def _maps(*paths):
    return "\n".join(f"7f00{i}000-7f00{i}fff r-xp 00000000 08:01 {i}  {p}" for i, p in enumerate(paths)) + "\n"


def _default_dir_find(maps):
    fake = types.ModuleType("torch")
    fake.__file__ = "/nonexistent/torch/__init__.py"
    saved = sys.modules.get("torch")
    sys.modules["torch"] = fake
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return hip_lib.find_hip_runtime(read_maps = lambda: maps), err.getvalue()
    finally:
        if saved is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = saved


@pytest.mark.skipif(hip_lib._WIN, reason = "no /proc/self/maps; Windows resolves the already-loaded DLL by name")
def test_default_dir_falls_back_to_the_mapped_library_and_logs_once():
    hip_lib._logged.clear()
    maps = _maps("/usr/lib/libc.so.6", "/venv/_rocm_sdk_core/lib/libamdhip64.so.7")
    path, log = _default_dir_find(maps)
    assert path == "/venv/_rocm_sdk_core/lib/libamdhip64.so.7"
    assert log.count("\n") == 1 and "already loaded" in log
    assert _default_dir_find(maps)[1] == ""


@pytest.mark.skipif(hip_lib._WIN, reason = "no /proc/self/maps; Windows resolves the already-loaded DLL by name")
def test_default_dir_none_when_nothing_is_mapped_or_maps_unreadable():
    hip_lib._logged.clear()
    assert _default_dir_find(_maps("/usr/lib/libc.so.6"))[0] is None

    def boom():
        raise OSError("no /proc")
    assert hip_lib._loaded_hip_runtime(boom) is None


def test_explicit_dir_never_uses_the_mapped_library(tmp_path):
    hip_lib._logged.clear()
    assert hip_lib.find_hip_runtime(str(tmp_path), read_maps = lambda: _maps("/x/libamdhip64.so.7")) is None


@pytest.mark.skipif(not hip_lib._WIN, reason = "POSIX finds the runtime through /proc/self/maps instead")
def test_win_candidates_scan_the_rocm_wheel_highest_first(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    bin_dir = site / "_rocm_sdk_core" / "bin"
    bin_dir.mkdir(parents = True)
    for n in ("amdhip64_7.dll", "amdhip64_10.dll", "amdhip64.bak"):
        (bin_dir / n).write_bytes(b"")
    fake = types.ModuleType("torch")
    fake.__file__ = str(site / "torch" / "__init__.py")
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert [os.path.basename(p) for p in hip_lib._win_candidates()] == ["amdhip64_10.dll", "amdhip64_7.dll"]


@pytest.mark.skipif(not hip_lib._WIN, reason = "POSIX finds the runtime through /proc/self/maps instead")
def test_win_default_dir_returns_the_bare_name_when_no_dll_is_visible(monkeypatch):
    # torch registers the wheel's DLL directory at import, so a bare module name is enough for ctypes
    # to bind the copy that is already loaded; it is reported once and never None
    hip_lib._logged.clear()
    monkeypatch.setattr(hip_lib, "_win_candidates", lambda: [])
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        path = hip_lib.find_hip_runtime()
        again = hip_lib.find_hip_runtime()
    assert path == again == "amdhip64"
    assert err.getvalue().count("\n") == 1 and "using amdhip64" in err.getvalue()
