"""Launcher configuration and engine arguments; deliberately GPU independent."""
import ipaddress
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = dict(family="qwen", model="", model_id="kyojin", host="127.0.0.1",
                port=8000, context=32768, sessions=1, vision=True, uncensor=False,
                cache_bits=0, draft_policy="mix", draft_tokens=7, drafter="",
                temperature="", top_p="", max_tokens="", reasoning="", thinking=True)
REASONING = {"qwen": ["", "low", "medium", "xhigh"],
             "glm": ["", "low", "medium", "high", "max"], "mimo": ["", "medium", "none"]}


def config_path():
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".config")) / "Kyojin" / "launcher.json"


def validate(raw, gui_port=8090, check_paths=False):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("Unknown settings or invalid settings object")
    cfg = DEFAULTS | raw
    if cfg["family"] not in REASONING:
        raise ValueError("Choose Qwen, GLM, or MiMo")
    for key in ("model", "model_id", "host", "drafter", "reasoning", "draft_policy"):
        if not isinstance(cfg[key], str) or "\0" in cfg[key]:
            raise ValueError(f"Invalid {key}")
    try:
        ipaddress.IPv4Address(cfg["host"])
    except ValueError:
        raise ValueError("API bind address must be an IPv4 address") from None
    for key, low, high in (("port", 1, 65535), ("context", 1, 1048576),
                           ("sessions", 1, 64), ("draft_tokens", 0, 32)):
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError(f"{key} must be a whole number from {low} to {high}")
    if cfg["port"] == gui_port:
        raise ValueError("API and control panel ports must differ")
    for key in ("vision", "uncensor", "thinking"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"Invalid {key}")
    if type(cfg["cache_bits"]) is not int or cfg["cache_bits"] not in (0, 8):
        raise ValueError("Cache bits must be 0 (automatic) or 8")
    if cfg["draft_policy"] not in ("mix", "mtp", "off"):
        raise ValueError("Invalid draft policy")
    if cfg["reasoning"] not in REASONING[cfg["family"]]:
        raise ValueError("Reasoning effort is not supported by this model family")
    if not cfg["model_id"].strip():
        raise ValueError("API model name is required")
    for key, low, high in (("temperature", 0, 10), ("top_p", 0, 1), ("max_tokens", 1, 1048576)):
        value = cfg[key]
        if value == "":
            continue
        try:
            number = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid {key}") from None
        if isinstance(value, bool) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        if key == "max_tokens" and not number.is_integer():
            raise ValueError("max_tokens must be a whole number")
        cfg[key] = int(number) if key == "max_tokens" else number
    if check_paths:
        pack = Path(cfg["model"]).expanduser()
        if not cfg["model"] or not pack.is_dir() or not (pack / "config.json").is_file():
            raise ValueError("Select an EXL3 model folder containing config.json")
        cfg["model"] = str(pack.resolve())
        if cfg["family"] == "mimo" and cfg["drafter"]:
            draft = Path(cfg["drafter"]).expanduser()
            if not draft.is_dir() or not (draft / "config.json").is_file():
                raise ValueError("Draft folder must contain config.json")
            cfg["drafter"] = str(draft.resolve())
    return cfg


def command(cfg, python, gui_port=8090, check_paths=True):
    c = validate(cfg, gui_port, check_paths)
    args = [str(python), "-u", str(ROOT / "tools" / c["family"] / "serve.py"),
            "--model", c["model"], "--model-id", c["model_id"], "--host", c["host"],
            "--port", str(c["port"]), "-c", str(c["context"])]
    if not c["vision"]:
        args.append("--no-vision")
    if not c["uncensor"]:
        args.append("--no-uncensor")
    for key, flag in (("temperature", "--default-temperature"), ("top_p", "--default-top-p"),
                      ("max_tokens", "--default-max-tokens"), ("reasoning", "--default-reasoning-effort")):
        if c[key] != "" and not (key == "reasoning" and c["family"] == "qwen" and not c["thinking"]):
            args.extend([flag, str(c[key])])
    if c["family"] in ("qwen", "glm"):
        args.extend(["--sessions", str(c["sessions"])])
    if c["family"] == "qwen":
        args.extend(["--cache-bits", str(c["cache_bits"]), "--draft-policy", c["draft_policy"],
                     "--ndt", str(c["draft_tokens"])])
        if not c["thinking"]:
            args.append("--no-thinking")
    elif c["family"] == "glm":
        args.extend(["--num-draft", str(c["draft_tokens"])])
    else:
        args.extend(["--ndt", str(c["draft_tokens"])])
        if c["draft_policy"] == "off":
            args.append("--no-dflash")
        if c["drafter"]:
            args.extend(["--drafter", c["drafter"]])
    return args


class Presets:
    def __init__(self, path, gui_port=8090):
        self.path, self.gui_port = Path(path), gui_port
        self.warning = ""
        self.data = {"selected": "Default", "presets": {"Default": DEFAULTS.copy()}}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data["presets"], dict) or not 1 <= len(data["presets"]) <= 100:
                    raise ValueError("Invalid preset collection")
                for name, cfg in data["presets"].items():
                    self.name(name)
                    validate(cfg, gui_port)
                if data["selected"] not in data["presets"]:
                    raise ValueError("Selected preset is missing")
                self.data = data
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.warning = f"Settings could not be read: {exc}. Original preserved; choose another config file to save."

    @staticmethod
    def name(name):
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError("Preset name must contain 1–80 characters")
        return name.strip()

    def save(self, name, cfg, delete=False):
        if self.warning:
            raise ValueError(self.warning)
        name = self.name(name)
        data = {"selected": name, "presets": self.data["presets"].copy()}
        if delete:
            if name not in data["presets"] or len(data["presets"]) == 1:
                raise ValueError("Keep at least one saved preset")
            del data["presets"][name]
            data["selected"] = next(iter(data["presets"]))
        else:
            data["presets"][name] = validate(cfg, self.gui_port)
            if len(data["presets"]) > 100:
                raise ValueError("At most 100 presets can be saved")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        self.data = data
