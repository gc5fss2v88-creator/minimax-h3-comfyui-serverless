#!/usr/bin/env bash
set -euo pipefail

models="${COMFYUI_PATH:-/comfyui}/models"
volume_root="${MODEL_VOLUME_PATH:-/runpod-volume}"
if [[ ! -d "$volume_root" ]]; then
  echo "[h3-feature] ERROR: Network Volume is not mounted at $volume_root" >&2
  exit 1
fi
mkdir -p "$volume_root/models" "$volume_root/profiling"
rm -rf "$models"
ln -s "$volume_root/models" "$models"
echo "[h3-feature] using persistent model volume at $volume_root"

H="${H3_MODEL_BASE_URL:-https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main}"
G="${H3_MXFP8_BASE_URL:-https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main}"
download() {
  local dir="$1" file="$2" url="$3"
  mkdir -p "$models/$dir"
  if [[ ! -s "$models/$dir/$file" ]]; then
    echo "[h3-feature] downloading $dir/$file"
    aria2c -x16 -s16 -k1M --console-log-level=warn --continue=true \
      --file-allocation=none --dir="$models/$dir" -o "$file" "$url"
  fi
}

# FL2VA and shared assets match the validated MXFP8 worker.
download diffusion_models minimax_h3_fl2va_mxfp8.safetensors "$G/minimax_h3_fl2va_mxfp8.safetensors"
# Ref2VA is the model required for Easy reference-video mode. Official ComfyUI
# publishes a pruned FP8 scaled variant compatible with the existing ComfyUI loader.
download diffusion_models minimax_h3_ref2va_pruned_fp8_scaled.safetensors "$H/diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
download text_encoders qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors "$H/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
download vae minimax_h3_video_vae_fp16.safetensors "$H/vae/minimax_h3_video_vae_fp16.safetensors"
download vae minimax_h3_audio_vae_fp32.safetensors "$H/vae/minimax_h3_audio_vae_fp32.safetensors"
download loras minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
download loras minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

/opt/venv/bin/python3 - <<'PY'
import torch
import comfy_kitchen
print(f"[h3-feature] MXFP8 runtime available; torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
try:
    import comfyui_minimaxh3_easy  # optional package name; node loading is checked below
except Exception:
    pass
PY

/opt/venv/bin/python3 - <<'PY'
import os, subprocess, sys, time, urllib.request
path = os.environ.get("COMFYUI_PATH", "/comfyui")
port = os.environ.get("COMFYUI_PORT", "8188")
args = [sys.executable, f"{path}/main.py", "--listen", "127.0.0.1", "--port", port]
extra = os.environ.get("COMFYUI_EXTRA_ARGS", "").split()
if os.environ.get("ENABLE_ATTENTION", "0") == "1":
    extra += os.environ.get("ATTENTION_ARGS", "").split()
args += extra
print("[h3-feature] starting ComfyUI", args, flush=True)
subprocess.Popen(args)
for _ in range(180):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2)
        print("[h3-feature] ComfyUI ready; Easy node pack requested", flush=True)
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("ComfyUI did not become ready")
PY

exec /opt/venv/bin/python3 /handler.py
