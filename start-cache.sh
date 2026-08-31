#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="/runpod-volume/models"
COMFY_ROOT="/comfyui/models"

mkdir -p "$COMFY_ROOT/diffusion_models"
mkdir -p "$COMFY_ROOT/text_encoders"
mkdir -p "$COMFY_ROOT/vae"

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

echo "H3 models found on Network Volume"
exec /start.sh
