#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python3}"
profile="${MODEL_PROFILE:-blackwell_fp8}"
volume_root="${MODEL_VOLUME_PATH:-/runpod-volume}"
models="${COMFYUI_PATH:-/comfyui}/models"
log() { printf '[h3-worker] %s\n' "$*"; }
fail() { log "FATAL: $*" >&2; exit 1; }
on_exit() { local rc=$?; [[ "$rc" -eq 0 ]] || log "worker startup exited with code=$rc" >&2; }
trap on_exit EXIT

log "startup pid=$$ profile=$profile attention=${ENABLE_ATTENTION:-0}"
log "python=$PYTHON_BIN version=$($PYTHON_BIN --version 2>&1)"
log "comfyui_path=${COMFYUI_PATH:-/comfyui} port=${COMFYUI_PORT:-8188}"
log "model_volume=$volume_root extra_args=${COMFYUI_EXTRA_ARGS:-<none>} attention_args=${ATTENTION_ARGS:-<none>}"
[[ -x "$PYTHON_BIN" ]] || fail "Python runtime not found: $PYTHON_BIN"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>&1 \
    | sed 's/^/[h3-worker] gpu=/' || log "nvidia-smi query failed"
else
  log "nvidia-smi=unavailable"
fi

if [[ "$profile" == "mxfp8_blackwell_candidate" || "$profile" == "blackwell_fp8" ]]; then
  models="${COMFYUI_PATH:-/comfyui}/models"
  if [[ -d "$volume_root" ]]; then
    mkdir -p "$volume_root/models"
    rm -rf "$models"
    ln -s "$volume_root/models" "$models"
    log "using persistent model volume at $volume_root"
  else
    fail "Network Volume is not mounted: $volume_root"
  fi
  require_model() {
    local dir="$1" file="$2" path="$models/$dir/$file"
    if [[ ! -s "$path" ]]; then
    log "ERROR: required model is missing: $path" >&2
    log "GPU startup downloads are disabled; populate the Network Volume with a CPU Pod" >&2
      exit 1
    fi
    log "model present: $dir/$file size=$(du -h "$path" | awk '{print $1}')"
  }
  if [[ "${MODEL_PROFILE:-blackwell_fp8}" == "mxfp8_blackwell_candidate" ]]; then
    require_model diffusion_models minimax_h3_fl2va_mxfp8.safetensors
  else
    require_model diffusion_models minimax_h3_fl2va_pruned_fp8_scaled.safetensors
  fi
  require_model text_encoders qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
  require_model vae minimax_h3_video_vae_fp16.safetensors
  require_model vae minimax_h3_audio_vae_fp32.safetensors
  require_model loras minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors
  require_model loras minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors
  "$PYTHON_BIN" - <<'PY'
import torch
import comfy_kitchen
print(f"[h3-worker] H3 model volume ready; torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
PY
fi

if [[ "${ENABLE_ATTENTION:-0}" == "1" && " ${ATTENTION_ARGS:-} " == *"--use-sage-attention"* ]]; then
  log "SageAttention preflight starting"
  "$PYTHON_BIN" - <<'PY'
import torch
import sageattention
from sageattention import sageattn

print(f"[h3-worker] SageAttention import ok: {sageattention}", flush=True)
print(f"[h3-worker] sageattn callable: {sageattn}", flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("SageAttention preflight: CUDA is unavailable")
device = torch.device("cuda")
print(f"[h3-worker] SageAttention GPU: {torch.cuda.get_device_name(device)} capability={torch.cuda.get_device_capability(device)}", flush=True)
q = torch.randn((1, 128, 8, 64), device=device, dtype=torch.float16)
k = torch.randn((1, 128, 8, 64), device=device, dtype=torch.float16)
v = torch.randn((1, 128, 8, 64), device=device, dtype=torch.float16)
out = sageattn(q, k, v, tensor_layout="HND")
torch.cuda.synchronize()
print(f"[h3-worker] SageAttention CUDA kernel preflight ok: output_shape={tuple(out.shape)}", flush=True)
PY
fi

"$PYTHON_BIN" - <<'PY'
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
proc = subprocess.Popen(args)
print(f"[h3-worker] ComfyUI pid={proc.pid}", flush=True)
for _ in range(180):
    rc = proc.poll()
    if rc is not None:
        raise SystemExit(f"ComfyUI exited before ready with returncode={rc}")
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2)
        print("[h3-worker] ComfyUI ready", flush=True)
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("ComfyUI did not become ready")
PY

log "starting Serverless handler"
exec "$PYTHON_BIN" /handler.py
