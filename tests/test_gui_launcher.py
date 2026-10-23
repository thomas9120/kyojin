"""GPU-free launcher checks, including real Windows process and HTTP lifetimes."""
import ast
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

GUI = Path(__file__).resolve().parents[1] / "tools" / "gui"
sys.path.insert(0, str(GUI))
from settings import DEFAULTS, ROOT, Presets, command, validate
from launcher_process import Engine
from server import Launcher


class SettingsTests(unittest.TestCase):
    def test_family_flags_match_actual_server_parsers(self):
        for family, effort in (("qwen", "xhigh"), ("glm", "max"), ("mimo", "none")):
            source = ast.parse((ROOT / "tools" / family / "serve.py").read_text(encoding="utf-8"))
            flags = {arg.value for node in ast.walk(source) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                     for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
            cfg = DEFAULTS | dict(family=family, model="C:/model path", reasoning=effort,
                                  temperature=0.5, top_p=0.8, max_tokens=200, vision=False)
            args = command(cfg, sys.executable, check_paths=False)
            self.assertIn("C:/model path", args)
            self.assertTrue(set(a for a in args[3:] if a.startswith("-")) <= flags)
            if family == "mimo":
                self.assertNotIn("--sessions", args)
                self.assertNotIn("--cache-bits", args)
            elif family == "glm":
                self.assertIn("--num-draft", args)
                self.assertNotIn("--draft-policy", args)

    def test_invalid_settings(self):
        for change in (dict(port=8090), dict(port=True), dict(context=0), dict(host="evil.example"),
                       dict(temperature="nan"), dict(top_p=2), dict(max_tokens=1.5),
                       dict(reasoning="max"), dict(vision="yes"), dict(extra="--anything")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(DEFAULTS | change)

    def test_missing_pack_and_qwen_thinking(self):
        with self.assertRaises(ValueError):
            command(DEFAULTS, sys.executable)
        args = command(DEFAULTS | dict(thinking=False, reasoning="xhigh"), sys.executable, check_paths=False)
        self.assertIn("--no-thinking", args)
        self.assertNotIn("--default-reasoning-effort", args)

    def test_presets_persist_delete_and_preserve_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "launcher.json"
            store = Presets(path)
            store.save("Second", DEFAULTS | dict(model="D:/path with spaces"))
            self.assertEqual(Presets(path).data["selected"], "Second")
            store.save("Default", None, delete=True)
            with self.assertRaises(ValueError):
                store.save("Second", None, delete=True)
            path.write_text("broken json", encoding="utf-8")
            broken = Presets(path)
            self.assertTrue(broken.warning)
            with self.assertRaises(ValueError):
                broken.save("Default", DEFAULTS)
            self.assertEqual(path.read_text(), "broken json")


class ProcessTests(unittest.TestCase):
    def test_loading_health_becomes_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "config.json").write_text("{}")
            engine = Engine(sys.executable, 8090)
            code = ("from http.server import BaseHTTPRequestHandler,HTTPServer\n"
                    "import time,json\n"
                    "started=time.monotonic()\n"
                    "class H(BaseHTTPRequestHandler):\n"
                    " def do_GET(self):\n"
                    "  ready=time.monotonic()-started>1\n"
                    "  self.send_response(200 if ready else 503);self.end_headers()\n"
                    "  self.wfile.write(json.dumps(dict(source='kyojin',status='ok' if ready else 'loading',message='Ready' if ready else 'Weights loading')).encode())\n"
                    "HTTPServer(('127.0.0.1',18092),H).serve_forever()")
            try:
                with patch("launcher_process.command", return_value=[sys.executable, "-u", "-c", code]):
                    engine.start(DEFAULTS | dict(model=temp, port=18092))
                states = set()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    state = engine.status()["state"]
                    states.add(state)
                    if state == "ready":
                        break
                    time.sleep(.05)
                self.assertEqual(states, {"loading", "ready"})
            finally:
                engine.stop()

    @unittest.skipUnless(sys.platform == "win32", "Windows Job Object")
    def test_launcher_crash_kills_owned_engine(self):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        code = ("import sys,subprocess,time;sys.path.insert(0,sys.argv[1]);"
                "from launcher_process import WindowsJob;job=WindowsJob();"
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],creationflags=4);"
                "job.assign(p);print(p.pid,flush=True);time.sleep(60)")
        parent = subprocess.Popen([sys.executable, "-u", "-c", code, str(GUI)], stdout=subprocess.PIPE,
                                  text=True, env=os.environ)
        handle = None
        try:
            pid = int(parent.stdout.readline())
            handle = kernel.OpenProcess(0x100000, False, pid)
            self.assertTrue(handle)
            parent.kill()
            parent.wait(timeout=5)
            self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            parent.stdout.close()
            if handle:
                kernel.CloseHandle(handle)

    def test_real_child_logs_duplicate_start_and_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            pack = Path(temp)
            (pack / "config.json").write_text("{}")
            engine = Engine(sys.executable, 8090)
            args = [sys.executable, "-u", "-c", "import time; print('test engine ready', flush=True); time.sleep(60)"]
            try:
                with patch("launcher_process.command", return_value=args):
                    engine.start(DEFAULTS | dict(model=temp, port=18091))
                    with self.assertRaises(ValueError):
                        engine.start(DEFAULTS | dict(model=temp, port=18091))
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    with engine.lock:
                        if "test engine ready" in engine.logs:
                            break
                    time.sleep(.05)
                self.assertIn("test engine ready", engine.logs)
                engine.stop()
                self.assertIsNotNone(engine.process.poll())
                engine.stop()  # idempotent
            finally:
                engine.stop()

    @unittest.skipUnless(sys.platform == "win32", "Windows Job Object")
    def test_job_kills_worker_when_engine_exits(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "config.json").write_text("{}")
            engine = Engine(sys.executable, 8090)
            code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); print(p.pid,flush=True);time.sleep(60)"
            try:
                with patch("launcher_process.command", return_value=[sys.executable, "-u", "-c", code]):
                    engine.start(DEFAULTS | dict(model=temp, port=18091))
                deadline = time.monotonic() + 5
                pid = None
                while time.monotonic() < deadline:
                    with engine.lock:
                        pids = [int(line) for line in engine.logs if line.isdigit()]
                    if pids:
                        pid = pids[0]
                        break
                    time.sleep(.05)
                self.assertIsNotNone(pid)
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.OpenProcess(0x100000, False, pid)
                self.assertTrue(handle)
                try:
                    engine.stop()
                    self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
                finally:
                    kernel.CloseHandle(handle)
            finally:
                engine.stop()


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = Launcher(0, Path(self.temp.name) / "launcher.json", sys.executable)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("GET" if body is None else "POST", path,
                           body=None if body is None else json.dumps(body), headers=headers or {})
        response = connection.getresponse()
        status, data = response.status, response.read()
        connection.close()
        return status, data

    def test_assets_security_and_save(self):
        for asset in ("/", "/app.js", "/style.css", "/api/status"):
            self.assertEqual(self.request(asset)[0], 200)
        status, data = self.request("/api/config")
        token = json.loads(data)["token"]
        body = dict(name="New", config=DEFAULTS)
        self.assertEqual(self.request("/api/save", body)[0], 403)
        headers = {"Origin": self.server.origin, "X-Launcher-Token": token}
        self.assertEqual(self.request("/api/save", body, headers)[0], 200)
        headers["Origin"] = "https://other.example"
        self.assertEqual(self.request("/api/stop", {}, headers)[0], 403)
        self.assertEqual(self.request("/api/config", headers={"Host": "other.example"})[0], 403)

    def test_browse_and_invalid_requests(self):
        from urllib.parse import quote
        (Path(self.temp.name) / "config.json").write_text("{}")
        status, body = self.request("/api/folders?path=" + quote(self.temp.name))
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["model"])
        headers = {"Origin": self.server.origin, "X-Launcher-Token": self.server.token}
        self.assertEqual(self.request("/api/start", {"config": DEFAULTS}, headers)[0], 400)
        self.assertEqual(self.request("/api/start", [], headers)[0], 400)


if __name__ == "__main__":
    unittest.main()
