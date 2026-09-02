#!/usr/bin/env bash
set -euo pipefail

echo "[h3-worker] profile=${MODEL_PROFILE:-blackwell_fp8} attention=${ENABLE_ATTENTION:-0}"
if [[ "${MODEL_PROFILE:-}" == "mxfp8_blackwell_candidate" ]]; then
  python3 - <<'PY'
import torch
import comfy_kitchen
print(f"[h3-worker] MXFP8 runtime available; torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
PY
fi
python3 - <<'PY'
import os, subprocess, sys, time, urllib.request
path = os.environ.get("COMFYUI_PATH", "/comfyui")
port = os.environ.get("COMFYUI_PORT", "8188")
args = [sys.executable, f"{path}/main.py", "--listen", "127.0.0.1", "--port", port]
extra = os.environ.get("COMFYUI_EXTRA_ARGS", "").split()
if os.environ.get("ENABLE_ATTENTION", "0") == "1":
    extra += os.environ.get("ATTENTION_ARGS", "").split()
    print("[h3-worker] attention args:", extra, flush=True)
args += extra
print("[h3-worker] starting ComfyUI", args, flush=True)
subprocess.Popen(args)
for _ in range(180):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2)
        print("[h3-worker] ComfyUI ready", flush=True)
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("ComfyUI did not become ready")
PY

exec python3 /handler.py
