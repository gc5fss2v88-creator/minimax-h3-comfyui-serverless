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

# Optional Ref2VA checkpoint. It is kept on the Network Volume and linked
# when present, so the normal FL2VA worker remains usable if it is absent.
if test -f "$MODEL_ROOT/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"; then
  ln -sf "$MODEL_ROOT/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
    "$COMFY_ROOT/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
  echo "Ref2VA model linked"
else
  echo "Ref2VA model not found (FL2VA remains available)"
fi

ln -sf "$MODEL_ROOT/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
"$COMFY_ROOT/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"

ln -sf "$MODEL_ROOT/vae/minimax_h3_video_vae_fp16.safetensors" \
"$COMFY_ROOT/vae/minimax_h3_video_vae_fp16.safetensors"

ln -sf "$MODEL_ROOT/vae/minimax_h3_audio_vae_fp32.safetensors" \
"$COMFY_ROOT/vae/minimax_h3_audio_vae_fp32.safetensors"

# Optional official Turbo LoRAs. They are stored separately on the Network
# Volume so the main H3 image stays small.
for lora in \
  minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
  minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors; do
  if test -f "$MODEL_ROOT/loras/$lora"; then
    ln -sf "$MODEL_ROOT/loras/$lora" "$COMFY_ROOT/loras/$lora"
    echo "LoRA linked: $lora"
  fi
done

echo "H3 models found on Network Volume"
exec /start.sh
