"""Loopback-only Kyojin control panel. Uses only the Python standard library."""
import argparse
import json
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from launcher_process import Engine
from settings import DEFAULTS, REASONING, Presets, command, config_path

STATIC = Path(__file__).parent / "static"


class Launcher(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, config, python):
        super().__init__(("127.0.0.1", port), Handler)
        self.token = secrets.token_urlsafe(32)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.presets = Presets(config, self.server_port)
        self.engine = Engine(python, self.server_port)
        self.edit_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, data, code=200, mime="application/json"):
        body = json.dumps(data).encode() if mime == "application/json" else data
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def allowed(self):
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

    def do_GET(self):
        if not self.allowed():
            return self.reply({"error": "Invalid host"}, 403)
        path = urlsplit(self.path).path
        if path == "/api/config":
            with self.server.edit_lock:
                return self.reply(dict(self.server.presets.data, token=self.server.token,
                                       warning=self.server.presets.warning, defaults=DEFAULTS, reasoning=REASONING))
        if path == "/api/status":
            return self.reply(self.server.engine.status())
        if path == "/api/folders":
            try:
                query = parse_qs(urlsplit(self.path).query)
                folder = Path(query.get("path", [str(Path.home())])[0]).expanduser().resolve()
                children = sorted((p for p in folder.iterdir() if p.is_dir()), key=lambda p: p.name.lower())
                return self.reply(dict(path=str(folder), parent=str(folder.parent),
                                       model=(folder / "config.json").is_file(),
                                       folders=[dict(name=p.name, path=str(p)) for p in children[:1000]]))
            except OSError as exc:
                return self.reply({"error": str(exc)}, 400)
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path not in files:
            return self.reply({"error": "Not found"}, 404)
        name, mime = files[path]
        self.reply((STATIC / name).read_bytes(), mime=mime)

    def do_POST(self):
        if (not self.allowed() or self.headers.get("Origin") != self.server.origin
                or not secrets.compare_digest(self.headers.get("X-Launcher-Token", ""), self.server.token)):
            return self.reply({"error": "Invalid origin or launcher token; reload this page"}, 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            path = urlsplit(self.path).path
            with self.server.edit_lock:
                if path == "/api/preview":
                    args = command(body["config"], self.server.engine.python, self.server.server_port, False)
                    return self.reply({"command": subprocess.list2cmdline(args)})
                if path in ("/api/save", "/api/delete"):
                    self.server.presets.save(body["name"], body.get("config"), delete=path.endswith("delete"))
                elif path == "/api/start":
                    self.server.engine.start(body["config"])
                elif path in ("/api/stop", "/api/exit"):
                    self.server.engine.stop()
                    if path == "/api/exit":
                        threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    return self.reply({"error": "Not found"}, 404)
            self.reply({"ok": True})
        except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError) as exc:
            self.reply({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--config", type=Path, default=config_path())
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    server = Launcher(args.port, args.config, sys.executable)
    # Windows console close/termination also closes the Job Object, killing workers.
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"Kyojin launcher: {server.origin}\nSettings: {args.config}\nKeep this terminal open.", flush=True)
    if not args.no_browser:
        webbrowser.open(server.origin)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        try:
            server.engine.stop()
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
