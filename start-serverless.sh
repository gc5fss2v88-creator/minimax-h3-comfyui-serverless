#!/usr/bin/env bash
set -euo pipefail

echo "[h3-worker] profile=${MODEL_PROFILE:-blackwell_fp8} attention=${ENABLE_ATTENTION:-0}"
if [[ "${MODEL_PROFILE:-blackwell_fp8}" == "mxfp8_blackwell_candidate" || "${MODEL_PROFILE:-blackwell_fp8}" == "blackwell_fp8" ]]; then
  models="${COMFYUI_PATH:-/comfyui}/models"
  volume_root="${MODEL_VOLUME_PATH:-/runpod-volume}"
  if [[ -d "$volume_root" ]]; then
    mkdir -p "$volume_root/models"
    rm -rf "$models"
    ln -s "$volume_root/models" "$models"
    echo "[h3-worker] using persistent model volume at $volume_root"
  fi
  require_model() {
    local dir="$1" file="$2" path="$models/$dir/$file"
    if [[ ! -s "$path" ]]; then
      echo "[h3-worker] ERROR: required model is missing: $path" >&2
      echo "[h3-worker] GPU startup downloads are disabled; pre-populate the Network Volume with a CPU Pod" >&2
      exit 1
    fi
    echo "[h3-worker] model present: $dir/$file"
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
  python3 - <<'PY'
import torch
import comfy_kitchen
print(f"[h3-worker] H3 model volume ready; torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
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
