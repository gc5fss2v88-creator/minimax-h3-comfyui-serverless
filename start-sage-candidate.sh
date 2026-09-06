#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python3}"
COMFYUI_PATH="${COMFYUI_PATH:-/comfyui}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
PROFILE="${MODEL_PROFILE:-mxfp8_blackwell_candidate}"
VOLUME_ROOT="${MODEL_VOLUME_PATH:-/runpod-volume}"
MODELS="${COMFYUI_PATH}/models"

log() { printf '[h3-sage] %s\n' "$*"; }
fatal() { log "FATAL: $*" >&2; exit 1; }

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    log "startup exited with code=$rc" >&2
  fi
}
trap on_exit EXIT

[[ -x "$PYTHON_BIN" ]] || fatal "Python runtime not found: $PYTHON_BIN"
log "startup pid=$$ profile=$PROFILE python=$PYTHON_BIN"
log "python_version=$($PYTHON_BIN --version 2>&1)"
log "comfyui_path=$COMFYUI_PATH port=$COMFYUI_PORT model_volume=$VOLUME_ROOT"

if [[ ! -d "$VOLUME_ROOT" ]]; then
  fatal "Network Volume is not mounted: $VOLUME_ROOT"
fi
mkdir -p "$VOLUME_ROOT/models"
rm -rf "$MODELS"
ln -s "$VOLUME_ROOT/models" "$MODELS"
log "using persistent model volume at $VOLUME_ROOT"

require_model() {
  local dir="$1"
  local file="$2"
  local path="$MODELS/$dir/$file"
  [[ -s "$path" ]] || fatal "required model missing: $path; GPU downloads are disabled"
  log "model present: $dir/$file size=$(du -h "$path" | awk '{print $1}')"
}

require_model diffusion_models minimax_h3_fl2va_mxfp8.safetensors
require_model text_encoders qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
require_model vae minimax_h3_video_vae_fp16.safetensors
require_model vae minimax_h3_audio_vae_fp32.safetensors
require_model loras minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors
require_model loras minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>&1 \
    | sed 's/^/[h3-sage] gpu=/' || log "nvidia-smi query failed"
else
  fatal "nvidia-smi is unavailable; Sage candidate requires a CUDA GPU"
fi

# This uses the same public entry point that ComfyUI 0.32 calls. Importing the
# package alone is insufficient because its CUDA extension is lazy-loaded.
"$PYTHON_BIN" - <<'PY'
import importlib.metadata
import torch
from sageattention import sageattn

if torch.version.cuda != "13.0":
    raise RuntimeError(f"expected PyTorch CUDA 13.0, got {torch.version.cuda}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
major, minor = torch.cuda.get_device_capability(0)
if (major, minor) != (12, 0):
    raise RuntimeError(f"expected Blackwell sm120, got sm{major}{minor}")

version = importlib.metadata.version("sageattention")
if version != "2.2.0":
    raise RuntimeError(f"expected SageAttention 2.2.0, got {version}")

# 2048 tokens exercises the same long-sequence path that H3 uses. The worker
# must fail before ComfyUI or RunPod starts accepting paid inference jobs.
q = torch.randn((1, 4, 2048, 64), device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
torch.cuda.synchronize()
if tuple(out.shape) != tuple(q.shape) or not torch.isfinite(out).all().item():
    raise RuntimeError(f"invalid SageAttention output: shape={tuple(out.shape)}")
print(
    f"[h3-sage] SageAttention CUDA kernel preflight ok; "
    f"package={version} torch={torch.__version__} cuda={torch.version.cuda} "
    f"arch=sm{major}{minor}",
    flush=True,
)
PY

"$PYTHON_BIN" - <<'PY'
import os
import subprocess
import sys
import threading
import time
import urllib.request

path = os.environ.get("COMFYUI_PATH", "/comfyui")
port = os.environ.get("COMFYUI_PORT", "8188")
args = [sys.executable, f"{path}/main.py", "--listen", "127.0.0.1", "--port", port]
args += os.environ.get("COMFYUI_EXTRA_ARGS", "").split()
if os.environ.get("ENABLE_ATTENTION", "1") == "1":
    args += os.environ.get("ATTENTION_ARGS", "--use-sage-attention").split()
else:
    raise SystemExit("Sage candidate requires ENABLE_ATTENTION=1")

print("[h3-sage] starting ComfyUI", args, flush=True)
proc = subprocess.Popen(
    args,
    cwd=path,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)


def forward_logs():
    assert proc.stdout is not None
    for line in proc.stdout:
        print("[comfyui] " + line.rstrip(), flush=True)

threading.Thread(target=forward_logs, name="comfyui-log-forwarder", daemon=True).start()
for _ in range(180):
    rc = proc.poll()
    if rc is not None:
        raise SystemExit(f"ComfyUI exited before ready with returncode={rc}")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2):
            print("[h3-sage] ComfyUI ready; SageAttention is selected", flush=True)
            break
    except Exception:
        time.sleep(1)
else:
    proc.terminate()
    raise SystemExit("ComfyUI did not become ready within 180 seconds")
PY

log "starting Serverless handler"
exec "$PYTHON_BIN" /handler.py
