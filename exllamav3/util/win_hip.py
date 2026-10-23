"""
Windows-only helpers for the HIP build path (native Windows + ROCm wheels).

Both entry points are no-ops on Linux and on CUDA-over-Windows, so importing this
module from setup.py / ext.py cannot change behaviour anywhere else.

1. patch_hipify()
   torch.utils.hipify compares the set of project sources against
   _to_unix_path(filepath). On Windows the set is built with backslashes, so every
   source falls into the "ignored" branch, hipified_path stays None and
   object_file_path(None) raises TypeError before the first compile. Normalizing
   the element type of that set is the whole fix; upstream-worthy, but a two-line
   monkeypatch here keeps the wheels we can actually install working.

2. maybe_set_device_lib_path()
   clang looks for the ROCm device library (ocml.bc, ockl.bc, hip.bc, ...) under
   $ROCM_PATH/amdgcn/bitcode, but the Windows ROCm wheels ship it at
   lib/llvm/amdgcn/bitcode, so every compile dies with
   "cannot find ROCm device library" until HIP_DEVICE_LIB_PATH is set. This is
   "TRAP 4" in tools/strix_halo/env.sh, and tools/windows/env.ps1 sets it too; this
   function is the fallback for shells that did not (the JIT builders compile at
   runtime, not just at install time).
"""
from __future__ import annotations

import os
import sys

__all__ = ["patch_hipify", "maybe_set_device_lib_path", "hip_toolchain_root", "win_link_libs"]


def patch_hipify() -> bool:
    """Make torch's hipify find its sources on Windows. Returns True if the patch is active."""
    if os.name != "nt":
        return False
    try:
        import torch
        if not torch.version.hip:
            return False
    except ImportError:
        return False
    try:
        from torch.utils.hipify import hipify_python
    except ImportError:
        return False
    if getattr(hipify_python, "_exl3_win_patched", False):
        return True

    original = hipify_python.preprocessor

    def preprocessor(output_directory, filepath, all_files, *args, **kwargs):
        if all_files is not None:
            all_files = [str(f).replace("\\", "/") for f in all_files]
        return original(output_directory, filepath, all_files, *args, **kwargs)

    hipify_python.preprocessor = preprocessor
    hipify_python._exl3_win_patched = True
    return True


def hip_toolchain_root() -> str | None:
    """Best guess at the ROCm devel tree holding headers, import libs and hipcc."""
    candidates = [
        os.environ.get("EXL3_ROCM_SDK"),
        os.environ.get("ROCM_HOME"),
        os.environ.get("HIP_PATH"),
        os.environ.get("ROCM_PATH"),
        os.path.join(sys.prefix, "Lib", "site-packages", "_rocm_sdk_devel"),
        os.path.join(sys.base_prefix, "Lib", "site-packages", "_rocm_sdk_devel"),
    ]
    for c in candidates:
        if not c:
            continue
        if os.path.isdir(os.path.join(c, "include", "hip")):
            return c
    return None


def _clang_ver_key(ver: str) -> tuple[int, ...]:
    """Numeric sort key for a clang resource-dir name ("17.0.6" -> (17, 0, 6)); names without digits sort first."""
    return tuple(int(p) for p in ver.split(".") if p.isdigit())


def win_link_libs() -> tuple[list[str], list[str]]:
    """
    (library_dirs, libraries) that the Windows link step needs on top of torch's own.

    Linux builds add no CUDA/HIP libraries at all: a shared object may keep symbols
    undefined and resolve them from libtorch_hip's own dependencies at load time. On Windows
    link.exe resolves everything up front, so the HIP BLAS entry points that hgemm.cu and
    graph.cu call directly need their import libraries. clang also lowers float->half
    conversions to __truncsfhf2, which libgcc provides on Linux and compiler-rt provides on
    Windows; nothing links it implicitly when the objects come from clang and the link goes
    through link.exe.
    """
    if os.name != "nt":
        return [], []
    root = hip_toolchain_root()
    if not root:
        return [], []
    libs = ["hipblas", "rocblas"]
    dirs = [os.path.join(root, "lib")]
    crt_root = os.path.join(root, "lib", "llvm", "lib", "clang")
    if os.path.isdir(crt_root):
        # Numeric, not lexicographic: "9.x" sorts above "22.x" as strings but is the older toolchain (F10)
        for ver in sorted(os.listdir(crt_root), key = _clang_ver_key, reverse = True):
            d = os.path.join(crt_root, ver, "lib", "windows")
            if os.path.isfile(os.path.join(d, "clang_rt.builtins-x86_64.lib")):
                dirs.append(d)
                libs.append("clang_rt.builtins-x86_64")
                break
    return dirs, libs


def maybe_set_device_lib_path() -> str | None:
    """Set HIP_DEVICE_LIB_PATH from the local toolchain if it is not already set."""
    if os.name != "nt":
        return None
    if os.environ.get("HIP_DEVICE_LIB_PATH"):
        return os.environ["HIP_DEVICE_LIB_PATH"]
    root = hip_toolchain_root()
    if not root:
        return None
    for rel in (("lib", "llvm", "amdgcn", "bitcode"), ("amdgcn", "bitcode")):
        path = os.path.join(root, *rel)
        if os.path.isdir(path):
            os.environ["HIP_DEVICE_LIB_PATH"] = path
            return path
    return None
