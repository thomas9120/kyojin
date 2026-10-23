"""One owned engine process, bounded logs, and crash-safe Windows tree cleanup."""
from collections import deque
import ctypes
from ctypes import wintypes
import json
import os
import socket
import subprocess
import threading
import urllib.error
import urllib.request

from settings import ROOT, command, validate


class WindowsJob:
    def __init__(self):
        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t),
                        ("max_ws", ctypes.c_size_t), ("active", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                        ("scheduling", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", ctypes.c_uint64 * 6),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        # Child was created suspended: assign before it can spawn any workers.
        resume = ctypes.WinDLL("ntdll").NtResumeProcess
        resume.argtypes = [wintypes.HANDLE]
        resume.restype = wintypes.LONG
        if resume(int(process._handle)) != 0:
            raise OSError("Could not resume the engine process")

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


class Engine:
    def __init__(self, python, gui_port):
        self.python, self.gui_port = python, gui_port
        self.lock = threading.RLock()
        self.logs = deque(maxlen=300)
        self.process = self.job = self.config = None
        self.args = []

    def _read(self, process):
        with process.stdout:
            for line in process.stdout:
                with self.lock:
                    self.logs.append(line.rstrip()[:8000])

    def start(self, config):
        with self.lock:
            if self.process and self.process.poll() is None:
                raise ValueError("The engine is already running; stop it before starting again")
            cfg = validate(config, self.gui_port, check_paths=True)
            args = command(cfg, self.python, self.gui_port)
            # Never treat an unrelated server on the same port as this engine.
            with socket.socket() as sock:
                try:
                    sock.bind((cfg["host"], cfg["port"]))
                except OSError:
                    raise ValueError("API port is already in use or cannot be bound") from None
            if self.job:
                self.job.close()
                self.job = None
            job = WindowsJob() if os.name == "nt" else None
            try:
                process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                           errors="replace", env=os.environ | {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                                           creationflags=0x4 if job else 0,
                                           start_new_session=os.name != "nt")
                try:
                    if job:
                        job.assign(process)
                except Exception:
                    process.kill()
                    process.wait(timeout=10)
                    process.stdout.close()
                    raise
            except Exception:
                if job:
                    job.close()
                raise
            self.process, self.job, self.config, self.args = process, job, cfg, args
            self.logs.clear()
            self.logs.append("Starting engine. First-launch kernel tuning can take several minutes.")
            threading.Thread(target=self._read, args=(process,), daemon=True).start()

    def stop(self):
        with self.lock:
            if self.job:
                self.job.terminate()
            elif self.process and self.process.poll() is None:
                import signal
                os.killpg(self.process.pid, signal.SIGKILL)
            if self.process:
                self.process.wait(timeout=15)
            if self.job:
                self.job.close()
                self.job = None

    def status(self):
        with self.lock:
            process, cfg = self.process, self.config
            result = dict(state="stopped", logs=list(self.logs), command=subprocess.list2cmdline(self.args),
                          running=cfg, api_url="", message="Engine is stopped")
            if not process:
                return result
            code = process.poll()
            if code is not None:
                result.update(state="exited", message=f"Engine exited (code {code})")
                return result
            host = "127.0.0.1" if cfg["host"] == "0.0.0.0" else cfg["host"]
            base = f"http://{host}:{cfg['port']}"
            result.update(state="loading", api_url=base + "/v1", message="Waiting for engine health endpoint…")
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                response = opener.open(base + "/health", timeout=1)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                health = json.loads(response.read(65536))
                if health.get("source") == "kyojin":
                    result["state"] = "ready" if response.code == 200 else health.get("status", "loading")
                    result["message"] = health.get("message") or ("Ready for requests" if response.code == 200 else "Loading model…")
                    result["progress"] = health.get("progress")
        except (OSError, ValueError):
            pass
        return result
