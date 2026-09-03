#!/usr/bin/env bash
set -euo pipefail

echo "[h3-worker] profile=${MODEL_PROFILE:-blackwell_fp8} attention=${ENABLE_ATTENTION:-0}"
if [[ "${MODEL_PROFILE:-}" == "mxfp8_blackwell_candidate" ]]; then
  models="${COMFYUI_PATH:-/comfyui}/models"
  volume_root="${MODEL_VOLUME_PATH:-/runpod-volume}"
  if [[ -d "$volume_root" ]]; then
    mkdir -p "$volume_root/models"
    rm -rf "$models"
    ln -s "$volume_root/models" "$models"
    echo "[h3-worker] using persistent model volume at $volume_root"
  fi
  H="${H3_MODEL_BASE_URL:-https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main}"
  G="${H3_MXFP8_BASE_URL:-https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main}"
  download() {
    local dir="$1" file="$2" url="$3"; mkdir -p "$models/$dir"
    if [[ ! -s "$models/$dir/$file" ]]; then
      echo "[h3-worker] downloading $dir/$file"
      aria2c -x16 -s16 -k1M --console-log-level=warn --dir="$models/$dir" -o "$file" "$url"
    fi
  }
  download diffusion_models minimax_h3_fl2va_mxfp8.safetensors "$G/minimax_h3_fl2va_mxfp8.safetensors"
  download text_encoders qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors "$H/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  download vae minimax_h3_video_vae_fp16.safetensors "$H/vae/minimax_h3_video_vae_fp16.safetensors"
  download vae minimax_h3_audio_vae_fp32.safetensors "$H/vae/minimax_h3_audio_vae_fp32.safetensors"
  download loras minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors"
  download loras minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
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
