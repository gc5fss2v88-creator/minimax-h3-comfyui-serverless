# MiniMax H3 Blackwell Serverless

RunPod Serverless worker for ComfyUI MiniMax H3 FL2VA/I2V, tuned for the RTX PRO 6000 Blackwell 96 GB profile.

The default model profile is **pruned FP8 scaled DiT + NVFP4-AWQ Qwen3-VL**. The worker keeps ComfyUI's native H3 nodes and exposes the ordinary ComfyUI API workflow contract. In ComfyUI Desktop, use **Save (API Format)** before sending a workflow.

## What is included

- `handler.py`: starts ComfyUI, uploads input images, queues an API workflow, polls history, and returns MP4 as a RunPod output artifact.
- `workflows/h3_i2v_api.json`: API-format I2V workflow with 15 s / 24 fps / 8-step defaults.
- `Dockerfile.cu130`: CUDA 13.0 runtime with a build-time `sm_120` guard for Blackwell.
- `.github/workflows/build.yml`: builds a slim image and appends large model files as remote layers with `crane`.
- `scripts/submit_runpod.py`: submit a local image and workflow to a RunPod endpoint.
- `scripts/benchmark.py`: runs one 20/8/6/4-step matrix; run it once per endpoint attention setting for a valid A/B comparison.

## Important model behavior

H3's native `MiniMaxH3ImageToVideo` node uses positive prompt conditioning and `BasicGuider`; it does not implement SD-style negative conditioning or CFG. The worker accepts `negative_prompt` and `cfg` in the request for a stable client schema, records them in metadata, and returns a warning when they are non-default. They are not silently wired into a node that does not support them.

For 6 steps, the worker uses the official 8-step Turbo LoRA because the official model repository publishes 4-step and 8-step FL2V LoRAs, not a distinct 6-step FL2V file. 6 steps is therefore a useful speed/quality operating point, but it is not a separately trained 6-step LoRA.

## Local checks

```bash
python3 -m unittest discover -s tests -v
python3 scripts/submit_runpod.py --endpoint-id "$RUNPOD_ENDPOINT_ID" --image input.png --prompt "A cinematic..."
```

## Build and deploy

The GitHub Action expects `GHCR_USERNAME` and `GHCR_TOKEN` only if you adapt it to a different registry. It publishes `ghcr.io/ORG/REPO:h3-blackwell-cu130-code` and then appends the FP8/NVFP4/VAE/LoRA layers.

RunPod settings for the final image:

| Field | Value |
|---|---|
| GPU | RTX PRO 6000 Blackwell 96 GB (`PRO_6000` / current RunPod name) |
| Container disk | 100 GB minimum; 120 GB preferred |
| Network volume | 100 GB minimum if weights are not baked; 0 GB when using the baked image |
| Workers | 0–1 during development, 1 production default |
| Idle timeout | 5–15 s |
| Execution timeout | 30 min for first cold start, then 15–20 min |
| FlashBoot | On |
| `MODEL_PROFILE` | `blackwell_fp8` |
| `COMFYUI_EXTRA_ARGS` | `--listen 127.0.0.1 --disable-auto-launch` |
| `COMFYUI_PORT` | `8188` |
| `ENABLE_ATTENTION` | `0` initially; enable only after the A/B benchmark passes |
| `ATTENTION_ARGS` | Empty initially; set only to a backend flag that is installed and validated in the image |

Use the immutable image digest in production after the first successful benchmark; do not deploy `:latest` for reproducibility.

## RunPod API input

```json
{
  "input": {
    "workflow": { "...": "ComfyUI API prompt JSON" },
    "images": [{"name": "input.png", "data": "<base64>"}],
    "params": {
      "prompt": "...", "negative_prompt": "", "width": 1344, "height": 768,
      "duration": 15, "fps": 24, "steps": 8, "seed": 42,
      "sampler": "res_multistep", "scheduler": "simple", "cfg": 1.0,
      "lora_strength": 1.0
    }
  }
}
```

The returned `output.video` is a base64 MP4 unless you configure the RunPod S3 output integration.

## MXFP8 candidate (isolated from baseline)

The separate `Dockerfile.mxfp8-candidate` and `build-mxfp8-candidate.yml` publish `:mxfp8-candidate`. They do not modify or retag the existing `:latest` FP8 scaled image. The candidate uses the full H3 FL2VA MXFP8 file `minimax_h3_fl2va_mxfp8.safetensors` from [rzgar/minimax_h3_fl2va_fp8_e4m3fn](https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn) plus ComfyUI `comfy-kitchen` MXFP8 tensor-core ops. The model card reports a 47.6 GB artifact and leaves quality-sensitive tensors at higher precision; compare quality against baseline before production use.

Create a second RunPod endpoint with image `ghcr.io/YOUR_ORG/minimax-h3-blackwell:mxfp8-candidate`, the same RTX PRO 6000 96 GB GPU, 120 GB container disk, 1 max worker, 10 s idle timeout, and 1800 s execution timeout. Set `MODEL_PROFILE=mxfp8_blackwell_candidate`, `ENABLE_ATTENTION=0`, and `COMFYUI_PORT=8188`. The existing FP8 endpoint remains untouched. Run the benchmark against both endpoints with identical input and seed.

Because this is the **full** MXFP8 artifact (about 47.6 GB), keeping both diffusion weights plus the shared encoder/VAEs on one persistent volume is about 90–92 GB before cache and outputs. A 100 GB volume can work only when the images are baked and the volume stores no download cache; 120 GB is the safer minimum, and 150 GB gives useful headroom.

## Attention gate

The image deliberately does not force SageAttention/FlashAttention at build time: Blackwell support and numerical behavior vary by torch/CUDA/backend release. Run the baseline with `ENABLE_ATTENTION=0`. For an attention candidate, build/install that backend in a separate image, set its ComfyUI flag through `ATTENTION_ARGS`, then run `scripts/benchmark.py --attention 1`. Compare the same seed and input against the baseline and inspect faces, hands, fast motion, camera motion, audio, and long clips. Enable it in production only after all checks pass; otherwise keep native attention.
