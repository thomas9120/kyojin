# Find torch's bundled HIP runtime. Stable wheels ship libamdhip64.so; nightly ROCm wheels ship only versioned
# names (libamdhip64.so.7, libamdhip64.so.7.1.2 ...). Plain name first, then the versioned ones, highest first.
# Windows: the ROCm wheel keeps amdhip64_7.dll outside torch/lib (in the _rocm_sdk_core package, whose directory
# torch registers with os.add_dll_directory at import), so the search covers those trees and a bare module name
# resolves against the directories torch already added.

import ctypes, glob, os, re, sys

_WIN = os.name == "nt"
_NAME = "amdhip64" if _WIN else "libamdhip64.so"
_DLL_SUFFIX = ".dll" if _WIN else ""
_logged = set()


def _version_key(path: str):
    return tuple(int(x) for x in re.findall(r"\d+", os.path.basename(path)[len(_NAME):]))


def _loaded_hip_runtime(read_maps) -> str | None:
    """Path of the libamdhip64 this process has already mapped (torch preloads it), or None."""
    try:
        lines = read_maps().splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split(None, 5)
        if len(parts) == 6 and os.path.basename(parts[5]).startswith(_NAME) and os.path.isabs(parts[5]):
            return parts[5]
    return None


def _read_proc_maps() -> str:
    with open("/proc/self/maps") as f:
        return f.read()


def _win_candidates() -> list[str]:
    """amdhip64*.dll in the trees a Windows ROCm wheel can put it in, highest version first."""
    import torch
    site = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
    pats = [os.path.join(site, "_rocm_sdk_core", "bin", _NAME + "*.dll"),
            os.path.join(site, "rocm_sdk_core", "bin", _NAME + "*.dll"),
            os.path.join(site, "torch", "lib", _NAME + "*.dll")]
    found = []
    for p in pats:
        found += [f for f in glob.glob(p) if os.path.isfile(f)]
    return sorted(found, key = _version_key, reverse = True)


def find_hip_runtime(lib_dir: str | None = None, read_maps = _read_proc_maps) -> str | None:
    """Path of libamdhip64 inside lib_dir (default: <torch>/lib), or None when there is none.

    With the default lib_dir, a torch wheel that keeps the library elsewhere (AMD nightly: _rocm_sdk_core/lib,
    preloaded at import) falls back to the copy this process has already mapped (read_maps: /proc/self/maps text).
    """
    default_dir = lib_dir is None
    if default_dir and _WIN:
        # No /proc/self/maps to consult. torch has already loaded the DLL and registered its
        # directory, so a bare name is enough for ctypes to bind the loaded module; prefer an
        # explicit path when one is visible, for the log line and for odd installs.
        found = _win_candidates()
        path = found[0] if found else _NAME
        if path not in _logged:
            _logged.add(path)
            print(f"hip_lib: using {path}", file = sys.stderr, flush = True)
        return path
    if default_dir:
        import torch
        lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    plain = os.path.join(lib_dir, _NAME)
    if os.path.isfile(plain):
        return plain
    versioned = [p for p in glob.glob(os.path.join(lib_dir, _NAME + ".*")) if os.path.isfile(p)
                 and re.fullmatch(r"(\.\d+)+", os.path.basename(p)[len(_NAME):])]
    if not versioned:
        loaded = _loaded_hip_runtime(read_maps) if default_dir else None
        if loaded and loaded not in _logged:
            _logged.add(loaded)
            print(f"hip_lib: {_NAME} not found in {lib_dir}, using the copy torch already loaded: {loaded}",
                  file = sys.stderr, flush = True)
        return loaded
    best = max(versioned, key = _version_key)
    if best not in _logged:
        _logged.add(best)
        print(f"hip_lib: {_NAME} not found in {lib_dir}, using {os.path.basename(best)}", file = sys.stderr, flush = True)
    return best


def load_hip_runtime(lib_dir: str | None = None) -> ctypes.CDLL:
    path = find_hip_runtime(lib_dir)
    if path is None:
        raise OSError(f"no {_NAME} or {_NAME}.<version> in {lib_dir or 'the torch lib directory'}")
    return ctypes.CDLL(path)
