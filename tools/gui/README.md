# Kyojin local launcher

A browser control panel for one native Kyojin inference server, following the
configure/save/start workflow of [Gufo for Windows](https://github.com/thomas9120/gufo-windows).
The launcher uses Python's standard library; it does not import torch or load a
model until you click **Start engine**.

## Install and launch

1. Install the AMD graphics driver, Python **3.12** (or `uv`), and Visual Studio
   2022/2026 Build Tools with **Desktop development with C++** and a Windows SDK.
   Hardware and manual setup: [Windows installation](../../doc/install-windows.md).
2. Double-click **setup-windows.bat** in the repository root. It creates `.venv`,
   installs the pinned Windows ROCm stack, extracts the HIP SDK, registers the
   package, builds the extension, and checks the GPU and extension import. Allow
   at least 45 GB free for setup, plus space for your model packs. Rerunning uses
   the existing environment and cached downloads. Logs are under
   `%LOCALAPPDATA%\Kyojin\logs`. Setup does not download models or install system
   drivers/Visual Studio. It stops with instructions if prerequisites are missing.
3. Double-click **launch-gui.bat**. It opens **http://127.0.0.1:8090** and leaves
   the launcher terminal open. Choose Qwen, GLM, or MiMo and browse to an existing
   **EXL3 pack folder containing config.json**. GGUF files are not supported.
4. Adjust settings, **Save** or **Save as…**, then **Start engine**. Start uses
   the current form without saving. Wait for **Ready**; the first kernel tuning
   can take several minutes. Connect your chat client to the displayed API URL
   (normally **http://127.0.0.1:8000/v1**) and model name (`kyojin` by default).

**Stop** unloads the engine and its workers. **Exit launcher & stop engine**
closes both. Closing the browser tab leaves the engine running. Ctrl+C in the
terminal stops both. On Windows the engine is assigned to a kill-on-close Job
Object before execution, so abrupt launcher termination also kills its process
tree. Stop interrupts active requests.

## Settings

Named presets live in `%LOCALAPPDATA%\Kyojin\launcher.json`. Saving is explicit;
the next visit restores the last saved preset without starting inference. Load
applies the selected preset to the form; Save updates the loaded preset. Delete
removes only a preset and keeps at least one. Invalid settings files remain
untouched; use `-Config` to choose another file before saving. Presets contain
absolute model paths and must be adjusted when moved to another machine.

The running model is shown separately from the next-launch form. Saving or
editing does not change the running server; stop and restart to apply changes.
Use a single editing tab: simultaneous saves from different tabs overwrite
earlier saves. The selected theme is stored only in the browser.

- All families expose context, API host/port/name, vision, bundled uncensor
  opt-in, temperature, top-p, output token limit, and family-specific reasoning.
  Blank generation fields use engine/model defaults; clients can override them.
- Qwen adds sessions, automatic/8-bit KV cache, thinking, and mixed/MTP/off
  speculative decoding. Context beyond 262K needs the 8-bit cache.
- GLM adds sessions and MTP draft tokens (`0` disables MTP). Choosing GLM sets
  draft depth to 2; Qwen/MiMo start at 7. Loading a preset restores its values.
- MiMo adds an optional DFlash draft pack folder. With speculation on and no
  explicit draft folder, the engine searches its normal model-relative locations.
  The Qwen MTP option is hidden for MiMo.
- Path checks require `config.json`; actual weights, draft/vision compatibility,
  GPU support, and available memory are validated by the engine. Errors appear
  in the engine log (last 300 lines).

The control panel always binds to `127.0.0.1`. Its mutation endpoints require
the same origin and a per-launch token. The engine API defaults to loopback;
choosing `0.0.0.0` exposes its unauthenticated API on your network. Remote clients
must use this PC's LAN address and the API port. The API and GUI ports must differ.

## Command-line options

```powershell
.\setup-windows.bat -Plan            # print setup steps without changes
.\setup-windows.bat -Jobs 4          # limit compile parallelism
.\setup-windows.bat -SkipBuild       # dependencies only; build before loading
.\launch-gui.bat -Port 8091
.\launch-gui.bat -NoBrowser -Config C:\Kyojin\launcher.json
```

The launch script reuses `tools/windows/env.ps1`, including `EXL3_VENV` and
`EXL3_ROCM_SDK` overrides. Setup targets this checkout's `.venv`.
Set `KYOJIN_NO_PAUSE=1` to suppress batch-script pauses in automation.
For development, the control panel can run without any engine dependencies:

```powershell
python tools/gui/server.py --no-browser --port 8090
python -m unittest discover -s tests -p test_gui_launcher.py -v
```

The tests use actual local HTTP and child processes, including Windows worker
cleanup. They require neither a GPU nor installed model weights.
