"""The tool files that exist once per server directory must stay identical.

`tools/{glm,mimo,qwen}/serve_ping.py` and friends were symlinks to a single implementation, which is
what kept the three servers on the same request log, /metrics text, image cache and SSE keep-alive.
Windows checkouts do not create symlinks without Developer Mode, and a checkout with
`core.symlinks=false` silently gets a text file whose content is the link target, so the port stores
them as real files. This test is what the symlink used to guarantee: an edit to one copy must be made
to all of them, and a silent divergence fails here instead of in one server's behaviour.

(Root `env.sh`, `bench_mtp.py` and `load_qwen.py` are the same story against `tools/strix_halo/`.)
"""
from __future__ import annotations

import filecmp
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SHARED_PER_SERVER = ("image_cache.py", "serve_metrics.py", "serve_ping.py", "serve_reqlog.py")
SERVERS = ("glm", "mimo", "qwen")
SHARED_ROOT = (("env.sh", "tools/strix_halo/env.sh"),
               ("bench_mtp.py", "tools/strix_halo/bench_mtp.py"),
               ("load_qwen.py", "tools/strix_halo/load_qwen.py"))


def _pairs():
    for name in SHARED_PER_SERVER:
        first = ROOT / "tools" / SERVERS[0] / name
        for server in SERVERS[1:]:
            yield first, ROOT / "tools" / server / name
    for a, b in SHARED_ROOT:
        yield ROOT / a, ROOT / b


PAIRS = list(_pairs())


@pytest.mark.parametrize("a,b", PAIRS, ids = [f"{a.name}~{b.parent.name}" for a, b in PAIRS])
def test_copies_are_identical(a: Path, b: Path):
    assert a.is_file() and b.is_file(), f"{a} or {b} is missing (this used to be a symlink)"
    assert filecmp.cmp(a, b, shallow = False), f"{a} and {b} diverged; edit both (they were one symlink)"


def test_no_shared_tool_path_is_still_a_symlink_entry():
    """A blob of mode 120000 with file *content* in it is what broke `git checkout` in CI: the entry
    asks for a symlink whose name is the whole file body ("File name too long")."""
    import subprocess

    out = subprocess.run(["git", "ls-files", "-s"], cwd = ROOT, capture_output = True, text = True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    symlinks = [line.split("\t", 1)[1] for line in out.stdout.splitlines() if line.startswith("120000 ")]
    assert symlinks == [], f"these are committed as symlinks: {symlinks}"
