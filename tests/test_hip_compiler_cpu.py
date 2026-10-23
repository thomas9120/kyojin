"""CPU tests of the shared JIT compiler resolver (exllamav3/util/hip_compiler.py) and of its use by the HIP loaders.
Fake hipcc scripts on a temp tree; no GPU, no ROCm, no built extension (the package __init__ files are bypassed
with empty namespace stubs while the modules are imported, then removed).

The fakes are real executables of the platform: a Python script with this interpreter as its shebang on POSIX,
and a .bat wrapper around the same script on Windows (CreateProcess cannot run a script directly). That keeps
the loader tests end to end - they run the fake through subprocess and check the file it writes."""
import importlib, os, stat, sys, types
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
_PKGS = ["exllamav3", "exllamav3.util", "exllamav3.vendor", "exllamav3.vendor.fla", "exllamav3.vendor.fla.hip",
         "exllamav3.modules", "exllamav3.modules.ple_fn", "exllamav3.modules.gr_fn"]
_added = []
for _n in _PKGS:
    if _n not in sys.modules:
        _m = types.ModuleType(_n); _m.__path__ = [str(ROOT.joinpath(*_n.split(".")))]
        sys.modules[_n] = _m; _added.append(_n)
try:
    hip_compiler = importlib.import_module("exllamav3.util.hip_compiler")
    LOADERS = {n: importlib.import_module(n) for n in (
        "exllamav3.modules.gr_fn.gr_mix_hip", "exllamav3.vendor.fla.hip.gdn_fused_h_hip", "exllamav3.vendor.fla.hip.kda_fused_h_hip",
        "exllamav3.modules.ple_fn.ple_hip")}
finally:
    for _n in _added:
        sys.modules.pop(_n, None)
_REAL_SDK_FROM_PACKAGE = hip_compiler._sdk_from_package


def _fake_body(version: str, log: Path | None, version_log: Path | None) -> str:
    """Python source of a fake compiler: --version answers, any other call writes the -o file as "built"."""
    lines = []
    if log:
        lines.append(f"open({str(log)!r}, 'a').write('built\\n')")
    if version_log:
        lines.append(f"open({str(version_log)!r}, 'a').write('called\\n')")
    return (
        "import sys\n"
        "a = sys.argv[1:]\n"
        "if '--version' in a:\n"
        f"    print({version!r})\n"
        + "".join(f"    {l}\n" for l in lines if version_log and l.startswith(f"open({str(version_log)!r}"))
        + "    raise SystemExit(0)\n"
        + "".join(f"{l}\n" for l in lines if log and l.startswith(f"open({str(log)!r}"))
        + "if '-o' in a:\n"
        "    open(a[a.index('-o') + 1], 'w').write('built\\n')\n"
    )


def fake_hipcc(path: Path, version: str, log: Path | None = None, version_log: Path | None = None) -> Path:
    """Create an executable fake compiler at `path` and return the file to use as the compiler."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _fake_body(version, log, version_log)
    if os.name == "nt":
        payload = path.with_name(path.name + "_fake.py")
        payload.write_text(body)
        shim = path.with_name(path.name + ".bat")       # hip_compiler._hipcc_in() looks for hipcc.bat
        shim.write_text(f'@echo off\r\n"{sys.executable}" "{payload}" %*\r\n')
        return shim
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k in ("EXL3_HIPCC", "EXL3_ROCM_SDK"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path / "emptybin"))
    # The HIP loaders cache their .hsaco in ~/.cache/exllamav3: point ~ at the temp tree so builds stay
    # hermetic (a leftover cache entry would turn the build into a hit and no fake compiler would run) and
    # never land in the developer's real cache. POSIX resolves ~ through HOME, Windows through USERPROFILE.
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home"), 1))
    monkeypatch.setattr(hip_compiler, "_sdk_from_package", lambda: None)
    hip_compiler.reset()
    yield tmp_path
    hip_compiler.reset()


def test_order_env_hipcc_first(env, monkeypatch):
    a = fake_hipcc(env / "a" / "hipcc", "clang A")
    sdk = env / "sdk"; sdk_cc = fake_hipcc(sdk / "bin" / "hipcc", "clang SDK")
    pkg = env / "pkg"; pkg_cc = fake_hipcc(pkg / "bin" / "hipcc", "clang PKG")
    path_cc = fake_hipcc(env / "pathbin" / "hipcc", "clang PATH")
    monkeypatch.setenv("PATH", str(env / "pathbin"))
    monkeypatch.setattr(hip_compiler, "_sdk_from_package", lambda: str(pkg))
    monkeypatch.setenv("EXL3_HIPCC", str(a)); monkeypatch.setenv("EXL3_ROCM_SDK", str(sdk))
    assert hip_compiler.resolve()["path"] == str(a)
    hip_compiler.reset(); monkeypatch.delenv("EXL3_HIPCC")
    assert hip_compiler.resolve()["path"] == str(sdk_cc)
    hip_compiler.reset(); monkeypatch.delenv("EXL3_ROCM_SDK")
    r = hip_compiler.resolve()
    assert r["path"] == str(pkg_cc) and r["source"] == "rocm-sdk package" and r["sdk_ok"]
    hip_compiler.reset(); monkeypatch.setattr(hip_compiler, "_sdk_from_package", lambda: None)
    r = hip_compiler.resolve()
    # which() on Windows reports the extension in PATHEXT's case (hipcc.BAT), the file we wrote is hipcc.bat
    assert os.path.normcase(r["path"]) == os.path.normcase(str(path_cc))
    assert r["source"] == "PATH" and not r["sdk_ok"]


def test_package_sdk_found_without_env(env, monkeypatch):
    # a fake site-packages with _rocm_sdk_devel next to the interpreter's purelib
    sp = env / "site-packages"
    cc = fake_hipcc(sp / "_rocm_sdk_devel" / "bin" / "hipcc", "clang SDK")
    monkeypatch.setattr(hip_compiler, "_sdk_from_package", _REAL_SDK_FROM_PACKAGE)
    import sysconfig
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(sp), "platlib": str(sp)})
    # only the fake tree: a real rocm-sdk in this venv would otherwise also be a candidate, and the
    # roots are compared as a set
    monkeypatch.setattr(sys, "path", [str(sp)])
    hip_compiler.reset()
    assert hip_compiler.resolve()["path"] == str(cc)
    assert hip_compiler.warning() is None


def test_none_found_is_safe(env, monkeypatch):
    monkeypatch.setattr(os.path, "isfile", lambda p: False)
    assert hip_compiler.resolve()["path"] is None
    assert "no hipcc found" in hip_compiler.warning()
    with pytest.raises(RuntimeError, match="hipcc not found"):
        hip_compiler.hipcc()


def test_key_depends_on_version(env, monkeypatch):
    a = fake_hipcc(env / "a" / "hipcc", "clang version 22")
    b = fake_hipcc(env / "b" / "hipcc", "clang version 23")
    monkeypatch.setenv("EXL3_HIPCC", str(a)); ka = hip_compiler.key()
    hip_compiler.reset(); monkeypatch.setenv("EXL3_HIPCC", str(b)); kb = hip_compiler.key()
    assert ka != kb and len(ka) == len(kb) == 8
    assert "clang version 23" in hip_compiler.describe() and str(b) in hip_compiler.describe()


def test_version_recorded_once(env, monkeypatch):
    log = env / "calls.log"
    a = fake_hipcc(env / "a" / "hipcc", "clang X", version_log = log)
    monkeypatch.setenv("EXL3_HIPCC", str(a))
    for _ in range(3):
        hip_compiler.key(); hip_compiler.describe(); hip_compiler.warning()
    assert log.read_text().count("called") == 1


def test_warning_text(env, monkeypatch):
    sdk = env / "sdk"
    fake_hipcc(sdk / "bin" / "hipcc", "clang 23")
    sysc = fake_hipcc(env / "opt" / "bin" / "hipcc", "clang 22")
    monkeypatch.setenv("EXL3_ROCM_SDK", str(sdk))
    assert hip_compiler.warning() is None
    hip_compiler.reset(); monkeypatch.delenv("EXL3_ROCM_SDK"); monkeypatch.setenv("EXL3_HIPCC", str(sysc))
    w = hip_compiler.warning()
    # FIX_LINE is the platform's export syntax (PowerShell on Windows)
    assert w.count("\n") == 0 and hip_compiler.FIX_LINE in w and "about 20 % slower" in w
    lines = []
    hip_compiler.report(lines.append)
    assert len(lines) == 2 and "HIP kernel compiler" in lines[0] and "WARNING" in lines[1]


@pytest.mark.parametrize("modname,call", [
    ("exllamav3.modules.gr_fn.gr_mix_hip", lambda m: m.compile_hsaco()),
    ("exllamav3.vendor.fla.hip.gdn_fused_h_hip", lambda m: m._compile("", "gfx1151")),
    ("exllamav3.vendor.fla.hip.kda_fused_h_hip", lambda m: m._compile("", "gfx1151")),
    ("exllamav3.modules.ple_fn.ple_hip", lambda m: m.compile_hsaco()),
])
def test_cache_key_includes_compiler(env, monkeypatch, modname, call):
    m = LOADERS[modname]
    log = env / "calls.log"
    a = fake_hipcc(env / "a" / "hipcc", "clang version 22", log)
    b = fake_hipcc(env / "b" / "hipcc", "clang version 23", log)
    monkeypatch.setenv("EXL3_HIPCC", str(a))
    pa = call(m)
    assert os.path.isfile(pa) and log.read_text().count("\n") == 1
    call(m)                                           # same compiler: cache hit, no second build
    assert log.read_text().count("\n") == 1
    # an "old" cache file (name without the compiler hash) is never reused: it is not the file the loader asks for
    hip_compiler.reset(); monkeypatch.setenv("EXL3_HIPCC", str(b))
    pb = call(m)
    assert pb != pa and log.read_text().count("\n") == 2 and open(pa).read() == "built\n"
    assert os.path.isfile(pa), "old files are not deleted"


def test_old_cache_name_not_reused(env, monkeypatch):
    import hashlib
    m = LOADERS["exllamav3.modules.gr_fn.gr_mix_hip"]
    log = env / "calls.log"
    a = fake_hipcc(env / "a" / "hipcc", "clang version 22", log)
    monkeypatch.setenv("EXL3_HIPCC", str(a))
    # recompute the pre-fix name (no compiler hash) and plant a stale slow binary there
    src = open(m._SRC, "rb").read()
    flags = ["--genco", "--offload-arch=gfx1151", "-O3", "-ffp-contract=off"]
    gcc = os.environ.get("EXL3_GCC_INSTALL_DIR", "/usr/lib/gcc/x86_64-linux-gnu/13")
    if os.path.isdir(gcc): flags.append(f"--gcc-install-dir={gcc}")
    # gr_mix_hip names its cache with sha1(<src> + flags [+ backslash-0 + compiler key])[:16]; the stale file has
    # the pre-fix name, without the compiler key. Not a security context: the digest only picks a filename
    name_hash = hashlib.sha1()
    name_hash.update(src + " ".join(flags).encode())
    old = Path(os.path.expanduser("~/.cache/exllamav3")) / f"gr_mix_gfx1151_{name_hash.hexdigest()[:16]}.hsaco"
    old.parent.mkdir(parents=True, exist_ok=True); old.write_text("slow")
    got = m.compile_hsaco()
    assert Path(got) != old and log.exists() and old.read_text() == "slow"


def test_servers_wired():
    for f in ("qwen", "glm", "mimo"):
        s = (ROOT / "tools" / f / "serve.py").read_text()
        assert "startup_health.report_compiler()" in s and "startup_health.compiler_health()" in s, f
    assert "hip_compiler.report" in (ROOT / "tools/strix_halo/env.sh").read_text()
    assert "Kernel compiler" in (ROOT / "tools/bench.py").read_text()


def test_compiler_health_fields(env, monkeypatch):
    # tools/ is not a package: loaded by name from the path above (a static import cannot be resolved here)
    sys.path.insert(0, str(ROOT / "tools"))
    startup_health = importlib.import_module("startup_health")
    for n in ("exllamav3", "exllamav3.util"):
        m = types.ModuleType(n); m.__path__ = [str(ROOT.joinpath(*n.split(".")))]
        monkeypatch.setitem(sys.modules, n, m)
    monkeypatch.setitem(sys.modules, "exllamav3.util.hip_compiler", hip_compiler)
    monkeypatch.setattr(sys.modules["exllamav3.util"], "hip_compiler", hip_compiler, raising=False)
    sysc = fake_hipcc(env / "opt" / "bin" / "hipcc", "clang 22")
    monkeypatch.setenv("EXL3_HIPCC", str(sysc))
    h = startup_health.compiler_health()
    assert "clang 22" in h["hip_compiler"] and "20 %" in h["hip_compiler_warning"]


def test_every_jit_site_uses_the_resolver():
    sites = ["exllamav3/modules/gr_fn/gr_mix_hip.py", "exllamav3/vendor/fla/hip/gdn_fused_h_hip.py", "exllamav3/vendor/fla/hip/kda_fused_h_hip.py",
             "exllamav3/vendor/fla/hip/gdn_hf_hip.py", "exllamav3/vendor/fla/hip/kda_hf_hip.py",
             "exllamav3/modules/ple_fn/ple_hip.py", "exllamav3/modules/attention_fn/qsa_prefill_hip.py",
                 "exllamav3/modules/dsv41_fn/dsv41_hip.py", "exllamav3/modules/dsv41_fn/dsv41_moe.py", "exllamav3/modules/dsv41_fn/ds41_pf.py"]
    for f in sites:
        s = (ROOT / f).read_text()
        assert "hip_compiler.hipcc()" in s and "hip_compiler.key()" in s and "EXL3_ROCM_SDK" not in s, f
    # every file that names --genco is one of those sites (a POSIX grep would need grep on PATH)
    hits = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "exllamav3").rglob("*.py")
                  if "--genco" in p.read_text(encoding = "utf-8", errors = "ignore"))
    assert hits == sorted(sites)
