import importlib.machinery
import os
import sys
import types

import pytest
import torch


def _install_extension_stub() -> None:
    """
    Register an empty `exllamav3_ext` module so that `import exllamav3` does not build the extension.

    Set EXL3_TEST_STUB_EXT=1 for this (CI's CPU-only job does). exllamav3/ext.py asks
    is_precompiled_extension_available() -- "does importlib find a module whose origin ends in an
    extension suffix?" -- and imports the module when the answer is yes; otherwise it starts a JIT
    build through torch.utils.cpp_extension, which needs a CUDA/HIP toolchain and minutes of compile.
    A stub in sys.modules wins over the file on disk and, with a matching __spec__.origin, takes the
    "precompiled" branch.

    The stub has no kernels, so any test that actually calls a binding fails with AttributeError --
    the right signal for "this one needs a real build". The CPU tests we run in that mode reach only
    pure Python (row cache, page table, spec gate, disk store, job scheduling).
    """
    name = "exllamav3_ext"
    if name in sys.modules:
        return                                  # a real build is importable: keep it
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    stub = types.ModuleType(name)
    stub.__doc__ = "CPU-test stub for the compiled extension (EXL3_TEST_STUB_EXT=1); no kernels."
    stub.__spec__ = importlib.machinery.ModuleSpec(
        name,
        importlib.machinery.ExtensionFileLoader(name, "exllamav3_ext_stub" + suffix),
        origin = "exllamav3_ext_stub" + suffix,
    )
    sys.modules[name] = stub


if os.environ.get("EXL3_TEST_STUB_EXT") == "1":
    _install_extension_stub()


@pytest.fixture(scope = "module")
def hide_gpus():
    """
    Hide every GPU for one test module (the ds41/dsv41 suites are CPU-only).

    Those modules used to set HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES at *import* time, and pytest
    imports every collected module before running anything -- so a plain `pytest tests/` hid the GPU
    from the engine tests that ran later in the same process. Setting the variables inside a fixture
    keeps the choice local to the module that asked for it, and restoring them afterwards keeps the
    rest of the session unaffected.
    """
    names = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    saved = {name: os.environ.get(name) for name in names}
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(autouse = True)
def _restore_current_cuda_device():
    # Several test modules pin tensors to "cuda:0" but launch Triton kernels on the *current* device,
    # while others call torch.cuda.set_device() for their own device and never restore it. Reset the
    # current device after every test so the suite's result doesn't depend on collection order
    yield
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
