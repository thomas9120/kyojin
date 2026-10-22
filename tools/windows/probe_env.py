#!/usr/bin/env python3
"""Windows bring-up probe for Kyojin (AMD gfx1151 / Strix Halo).

Answers the four questions the port plan gates on (doc/windows-port.md, section 4):

  A. how much GPU-visible memory this process can really get
  B. whether the ROCm-on-Windows toolchain (hipcc + headers + bitcode) is usable
  C. whether a kernel compiled at runtime (--genco) loads and runs
  D. rough compute throughput, for a first sanity number

Run it inside the venv that has the ROCm torch wheel:

    py -3.12 -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install <rocm_sdk_* wheels> <torch wheel>
    python tools/windows/probe_env.py

It also works on Linux, which makes the Windows numbers directly comparable.
Nothing here is imported by the engine; it is a standalone diagnostic.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

GIB = 1 << 30
MIB = 1 << 20

report: dict = {}


def say(msg: str = ""):
    print(msg, flush = True)


def head(title: str):
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def put(key: str, value):
    report[key] = value
    say(f"  {key}: {value}")


# ---------------------------------------------------------------------------------------------
# torch / device
# ---------------------------------------------------------------------------------------------

def probe_torch():
    head("torch")
    import torch
    put("python", sys.version.split()[0])
    put("torch", torch.__version__)
    put("torch.version.hip", torch.version.hip)
    put("torch.version.cuda", torch.version.cuda)
    put("platform", sys.platform)
    put("torch/lib", os.path.join(os.path.dirname(torch.__file__), "lib"))
    put("cuda.is_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        say("  !! no GPU: the memory/JIT sections will be skipped")
        return torch, None
    n = torch.cuda.device_count()
    put("device_count", n)
    p = torch.cuda.get_device_properties(0)
    gcn = getattr(p, "gcnArchName", None)
    put("device[0]", p.name)
    put("gcnArchName", gcn)
    put("total_memory_GiB", round(p.total_memory / GIB, 2))
    put("multi_processor_count", getattr(p, "multi_processor_count", None))
    put("warp_size", getattr(p, "warp_size", None))
    put("is_integrated", getattr(p, "is_integrated", None))
    try:
        free, total = torch.cuda.mem_get_info()
        put("mem_get_info_free_GiB", round(free / GIB, 2))
        put("mem_get_info_total_GiB", round(total / GIB, 2))
    except Exception as e:  # noqa: BLE001
        put("mem_get_info", f"failed: {e}")
    return torch, p


def probe_memory(torch, single_max: int, aggregate_max: int):
    head("A. memory ladder (single allocations)")
    say("  Windows caps the HIP SVM pool at ~64 GiB on gfx1151 until the CLR/driver fix lands;")
    say("  see doc/windows-port.md. If a step fails, note it and try OCL_SET_SVM_SIZE=262144.")
    ladder = [g for g in (2, 4, 8, 16, 32, 48, 56, 60, 64, 72, 80, 88, 96, 104, 112) if g <= single_max]
    single = None
    for g in ladder:
        try:
            t = torch.empty(g * GIB, dtype = torch.uint8, device = "cuda")
            t.zero_()                          # touch: forces the pages to be backed
            torch.cuda.synchronize()
            del t
            torch.cuda.empty_cache()
            single = g
            say(f"  {g:>4} GiB  ok")
        except (RuntimeError, MemoryError) as e:
            say(f"  {g:>4} GiB  FAIL ({str(e).splitlines()[0][:120]})")
            torch.cuda.empty_cache()
            break
    put("largest_single_allocation_GiB", single)

    head("A. aggregate ceiling (many 4 GiB allocations, touched)")
    chunks = []
    got = 0
    step = 4
    try:
        while got < aggregate_max:
            t = torch.empty(step * GIB, dtype = torch.uint8, device = "cuda")
            t.zero_()
            chunks.append(t)
            got += step
            say(f"  {got:>4} GiB  ok")
    except (RuntimeError, MemoryError) as e:
        say(f"  {got + step:>4} GiB  FAIL ({str(e).splitlines()[0][:120]})")
    put("aggregate_ceiling_GiB", got)
    free, total = (0, 0)
    try:
        free, total = torch.cuda.mem_get_info()
        put("mem_get_info_at_ceiling_GiB", (round(free / GIB, 2), round(total / GIB, 2)))
    except Exception:  # noqa: BLE001
        pass
    del chunks
    torch.cuda.empty_cache()


def probe_compute(torch):
    head("D. compute sanity")
    try:
        a = torch.randn(4096, 4096, dtype = torch.float16, device = "cuda")
        b = torch.randn(4096, 4096, dtype = torch.float16, device = "cuda")
        for _ in range(3):
            c = a @ b
        torch.cuda.synchronize()
        n = 3
        t0 = time.perf_counter()
        for _ in range(n):
            c = a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
        tflops = 2 * 4096 ** 3 / dt / 1e12
        put("fp16_4096_matmul_TFLOPs", round(tflops, 2))
        # decode-shaped: one row against a big matrix (what the engine does per token)
        m = torch.randn(8192, 8192, dtype = torch.float16, device = "cuda")
        x = torch.randn(1, 8192, dtype = torch.float16, device = "cuda")
        for _ in range(5):
            y = x @ m
        torch.cuda.synchronize()
        n = 20
        t0 = time.perf_counter()
        for _ in range(n):
            y = x @ m
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
        put("gemv_8192x8192_GB_per_s", round(m.numel() * 2 / dt / 1e9, 2))
        say("  (Linux gfx1151 reference: ~31 TFLOP/s fp16 4096^3)")
    except Exception as e:  # noqa: BLE001
        put("compute", f"failed: {e}")


# ---------------------------------------------------------------------------------------------
# toolchain
# ---------------------------------------------------------------------------------------------

def _sdk_from_package() -> str | None:
    """_rocm_sdk_devel inside the running interpreter, if the wheel is installed."""
    import sysconfig
    roots = {sysconfig.get_paths().get(k) for k in ("purelib", "platlib")}
    roots |= {p for p in sys.path if p.endswith("site-packages")}
    for r in sorted(x for x in roots if x):
        d = os.path.join(r, "_rocm_sdk_devel")
        if os.path.isdir(d):
            return d
    tool = os.path.join(os.path.dirname(sys.executable), "rocm-sdk")
    for exe in (tool, tool + ".exe"):
        if os.path.isfile(exe):
            try:
                out = subprocess.run([exe, "path", "--root"], capture_output = True, text = True,
                                     timeout = 30).stdout.strip()
                if out and os.path.isdir(out):
                    return out
            except Exception:  # noqa: BLE001
                pass
    return None


def _hipcc_candidates():
    names = ("hipcc.exe", "hipcc.bat", "hipcc.cmd", "hipcc") if os.name == "nt" else ("hipcc",)
    roots = [os.environ.get("ROCM_HOME"), os.environ.get("ROCM_PATH"),
             os.environ.get("HIP_PATH"), _sdk_from_package(),
             os.environ.get("HIP_SDK_PATH")]
    out = []
    if env := os.environ.get("EXL3_HIPCC"):
        out.append(("EXL3_HIPCC", env))
    for r in roots:
        if not r:
            continue
        for n in names:
            out.append((r, os.path.join(r, "bin", n)))
    if found := shutil.which("hipcc"):
        out.append(("PATH", found))
    out.append(("/opt/rocm", "/opt/rocm/bin/hipcc"))
    return out


def probe_toolchain():
    head("B. toolchain")
    sdk = _sdk_from_package()
    put("_rocm_sdk_devel", sdk)
    put("ROCM_HOME", os.environ.get("ROCM_HOME"))
    put("HIP_PATH", os.environ.get("HIP_PATH"))

    hipcc = None
    for src, p in _hipcc_candidates():
        if p and os.path.isfile(p):
            say(f"  hipcc candidate ({src}): {p}")
            if hipcc is None:
                hipcc = p
    put("hipcc", hipcc)
    if hipcc:
        try:
            r = subprocess.run([hipcc, "--version"], capture_output = True, text = True, timeout = 120)
            ver = " ".join((r.stdout or r.stderr).split())[:400]
            put("hipcc --version", ver)
            m = re.search(r"(clang version [\d.]+)", ver)
            if m:
                put("clang", m.group(1))
        except Exception as e:  # noqa: BLE001
            put("hipcc --version", f"failed: {e}")

    # headers and device bitcode: the Linux build needs hipsparse/ and thrust/ (see tools/strix_halo/env.sh)
    for label, root in (("sdk", sdk), ("ROCM_HOME", os.environ.get("ROCM_HOME"))):
        if not root:
            continue
        checks = {
            "hipsparse": ["include/hipsparse/hipsparse.h", "include/hipsparse.h"],
            "thrust": ["include/thrust/version.h"],
            "hipcub": ["include/hipcub/hipcub.hpp"],
            "hip/hip_runtime.h": ["include/hip/hip_runtime.h"],
            "bitcode(llvm)": ["lib/llvm/amdgcn/bitcode/ocml.bc"],
            "bitcode(amdgcn)": ["amdgcn/bitcode/ocml.bc"],
        }
        say(f"  -- {label}: {root}")
        for name, rels in checks.items():
            hit = next((r for r in rels if os.path.isfile(os.path.join(root, r.replace("/", os.sep)))), None)
            put(f"header {name} ({label})", hit or "MISSING")
    put("ocml bitcode via HIP_DEVICE_LIB_PATH", os.environ.get("HIP_DEVICE_LIB_PATH"))


def _hip_dll() -> str | None:
    import torch
    libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
    pats = ["amdhip64*.dll", "libamdhip64.so*"] if os.name == "nt" else ["libamdhip64.so*"]
    hits = []
    for pat in pats:
        hits += glob.glob(os.path.join(libdir, pat))
    if not hits:
        return None
    return max(hits, key = lambda p: [int(x) for x in re.findall(r"\d+", os.path.basename(p))] or [0])


_JIT_SRC = r"""
#include <hip/hip_runtime.h>
extern "C" __global__ void vadd(const float* a, const float* b, float* c)
{
    int i = threadIdx.x;
    if (i < 256) c[i] = a[i] + b[i];
}
"""


def probe_jit(torch, arch: str):
    head("C. JIT: --genco -> hipModuleLoad -> launch")
    hipcc = next((p for _, p in _hipcc_candidates() if p and os.path.isfile(p)), None)
    if not hipcc:
        put("jit", "skipped: no hipcc")
        return
    tmp = tempfile.mkdtemp(prefix = "kyojin-probe-")
    src = os.path.join(tmp, "vadd.hip")
    out = os.path.join(tmp, "vadd.hsaco")
    with open(src, "w") as f:
        f.write(_JIT_SRC)
    cmd = [hipcc, "--genco", f"--offload-arch={arch}", "-O3", "-o", out, src]
    say("  compile: " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output = True, text = True)
    if r.returncode != 0:
        put("jit_compile", "FAIL")
        say("  stderr:")
        say("  " + (r.stderr or "")[-4000:].replace("\n", "\n  "))
        return
    put("jit_compile", f"ok ({os.path.getsize(out)} bytes)")

    dll = _hip_dll()
    put("hip runtime dll", dll)
    if not dll:
        put("jit_load", "skipped: no hip runtime next to torch")
        return
    try:
        lib = ctypes.CDLL(dll)
        mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        if lib.hipModuleLoad(ctypes.byref(mod), out.encode()) != 0:
            put("jit_load", "hipModuleLoad failed")
            return
        if lib.hipModuleGetFunction(ctypes.byref(fn), mod, b"vadd") != 0:
            put("jit_load", "hipModuleGetFunction failed")
            return
        a = torch.arange(256, dtype = torch.float32, device = "cuda")
        b = torch.ones(256, dtype = torch.float32, device = "cuda")
        c = torch.zeros(256, dtype = torch.float32, device = "cuda")
        args = [ctypes.c_void_p(t.data_ptr()) for t in (a, b, c)]
        params = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in args])
        lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        rc = lib.hipModuleLaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, None, params, None)
        torch.cuda.synchronize()
        ok = rc == 0 and bool(torch.allclose(c, a + b))
        put("jit_launch", "ok (vadd correct)" if ok else f"bad (rc={rc})")
        lib.hipModuleUnload(mod)
    except Exception as e:  # noqa: BLE001
        put("jit_launch", f"failed: {e}")


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single-max", type = int, default = 112, help = "largest single allocation to try (GiB)")
    ap.add_argument("--aggregate-max", type = int, default = 120, help = "stop the aggregate ladder here (GiB)")
    ap.add_argument("--no-memory", action = "store_true", help = "skip the allocation ladders")
    ap.add_argument("--json", default = None, help = "write the collected numbers here")
    args = ap.parse_args()

    head("environment")
    for k in ("OCL_SET_SVM_SIZE", "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
              "PYTORCH_ROCM_ARCH", "EXL3_HIPCC", "HIP_DEVICE_LIB_PATH", "PYTORCH_CUDA_ALLOC_CONF"):
        say(f"  {k} = {os.environ.get(k)!r}")

    torch, props = probe_torch()
    arch = "gfx1151"
    if props is not None:
        gcn = (getattr(props, "gcnArchName", "") or "").split(":")[0]
        if gcn.startswith("gfx"):
            arch = gcn
    put("arch", arch)

    if torch.cuda.is_available() and not args.no_memory:
        probe_memory(torch, args.single_max, args.aggregate_max)
        probe_compute(torch)
    probe_toolchain()
    if torch.cuda.is_available():
        probe_jit(torch, arch)

    head("summary")
    for k, v in report.items():
        say(f"  {k}: {v}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent = 2, default = str)
        say(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
