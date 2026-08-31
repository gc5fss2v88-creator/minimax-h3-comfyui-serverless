#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="/runpod-volume/huggingface-cache/hub/models--Comfy-Org--MiniMax-H3/snapshots"
SNAPSHOT="$(find "$CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d | head -n 1)"

if [ -z "$SNAPSHOT" ]; then
  echo "ERROR: MiniMax H3 Cached Model snapshot not found"
  exit 1
fi

mkdir -p /comfyui/models/diffusion_models
mkdir -p /comfyui/models/text_encoders
mkdir -p /comfyui/models/vae

ln -sf "$SNAPSHOT/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  /comfyui/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors

ln -sf "$SNAPSHOT/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  /comfyui/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors

ln -sf "$SNAPSHOT/vae/minimax_h3_video_vae_fp16.safetensors" \
  /comfyui/models/vae/minimax_h3_video_vae_fp16.safetensors

ln -sf "$SNAPSHOT/vae/minimax_h3_audio_vae_fp32.safetensors" \
  /comfyui/models/vae/minimax_h3_audio_vae_fp32.safetensors

exec /start.sh
