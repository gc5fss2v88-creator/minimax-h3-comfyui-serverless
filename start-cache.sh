#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="/runpod-volume/models"
COMFY_ROOT="/comfyui/models"

mkdir -p "$COMFY_ROOT/diffusion_models"
mkdir -p "$COMFY_ROOT/text_encoders"
mkdir -p "$COMFY_ROOT/vae"
mkdir -p "$COMFY_ROOT/loras"

test -f "$MODEL_ROOT/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
test -f "$MODEL_ROOT/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
test -f "$MODEL_ROOT/vae/minimax_h3_video_vae_fp16.safetensors"
test -f "$MODEL_ROOT/vae/minimax_h3_audio_vae_fp32.safetensors"

ln -sf "$MODEL_ROOT/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
"$COMFY_ROOT/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"

ln -sf "$MODEL_ROOT/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
"$COMFY_ROOT/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"

ln -sf "$MODEL_ROOT/vae/minimax_h3_video_vae_fp16.safetensors" \
"$COMFY_ROOT/vae/minimax_h3_video_vae_fp16.safetensors"

ln -sf "$MODEL_ROOT/vae/minimax_h3_audio_vae_fp32.safetensors" \
"$COMFY_ROOT/vae/minimax_h3_audio_vae_fp32.safetensors"

# Optional Turbo LoRA. It is uploaded separately to the Network Volume so the
# main H3 model image stays small. ComfyUI will list it when present.
if test -f "$MODEL_ROOT/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"; then
  ln -sf "$MODEL_ROOT/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" \
    "$COMFY_ROOT/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
fi

echo "H3 models found on Network Volume"
exec /start.sh
