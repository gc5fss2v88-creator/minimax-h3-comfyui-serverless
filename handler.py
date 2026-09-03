"""RunPod handler that adapts the ComfyUI HTTP API to Serverless jobs."""
import base64, copy, json, os, pathlib, time, uuid
from typing import Any


COMFY = f"http://127.0.0.1:{os.getenv('COMFYUI_PORT', '8188')}"
TEMPLATE = pathlib.Path(os.getenv('WORKFLOW_TEMPLATE', str(pathlib.Path(__file__).parent / 'workflows/h3_i2v_api.json')))
DEFAULTS = {"width": 1344, "height": 768, "duration": 15, "fps": 24, "steps": 8,
            "seed": 42, "sampler": "res_multistep", "scheduler": "simple",
            "cfg": 1.0, "lora_strength": 1.0, "prompt": "", "negative_prompt": ""}
MODEL = os.getenv("MODEL_PROFILE", "blackwell_fp8")
LORAS = {4: os.getenv("LORA_4STEP", "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors"),
         8: os.getenv("LORA_8STEP", "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors")}

def _json(data):
    return json.dumps(data, ensure_ascii=False)

def _params(inp):
    p = {**DEFAULTS, **(inp.get("params") or {})}
    if p["steps"] not in (4, 6, 8, 20):
        raise ValueError("steps must be one of 4, 6, 8, 20")
    if not (1 <= int(p["duration"]) <= 15):
        raise ValueError("duration must be between 1 and 15 seconds")
    if int(p["fps"]) != 24:
        raise ValueError("H3 FL2VA currently supports 24 fps only")
    p["width"], p["height"] = int(p["width"]), int(p["height"])
    if p["width"] % 32 or p["height"] % 32 or p["width"] * p["height"] > 1344 * 768:
        raise ValueError("width/height must be multiples of 32 and area <= 1344x768")
    if not p["prompt"]:
        raise ValueError("prompt is required")
    return p

def _frames(seconds):
    # H3 uses a 17k+5 latent frame grid; these are the valid 24fps approximations.
    return max(17, int(round(seconds * 24 / 17)) * 17 + 5)

def _workflow(inp, p):
    wf = copy.deepcopy(inp.get("workflow") or json.loads(TEMPLATE.read_text()))
    # Keep node ids stable in the shipped API template; exported Desktop workflows can
    # also be used when they retain these semantic node types and fields.
    by_type = {v.get("class_type"): (k, v) for k, v in wf.items() if isinstance(v, dict)}
    def node(kind):
        if kind not in by_type:
            raise ValueError(f"workflow is missing required node: {kind}")
        return by_type[kind][1]["inputs"]
    cond = node("MiniMaxH3ImageToVideo")
    cond.update({"prompt": p["prompt"], "width": p["width"], "height": p["height"], "length": _frames(p["duration"])})
    node("RandomNoise")["noise_seed"] = int(p["seed"])
    node("KSamplerSelect")["sampler_name"] = p["sampler"]
    node("BasicScheduler").update({"steps": int(p["steps"]), "scheduler": p["scheduler"]})
    node("CreateVideo")["fps"] = int(p["fps"])
    # 6 steps has no separately trained official FL2V LoRA; 8-step is the closest
    # supported Turbo operating point.  20 is the base/reference benchmark.
    for _, v in wf.items():
        if isinstance(v, dict) and v.get("class_type") in ("LoraLoaderModelOnly", "LoraLoader"):
            v["inputs"]["lora_name"] = LORAS.get(int(p["steps"]), "")
            v["inputs"]["strength_model"] = float(p["lora_strength"])
    if int(p["steps"]) in (4, 6, 8):
        chosen = LORAS[4] if int(p["steps"]) == 4 else LORAS[8]
        wf.setdefault("lora", {"class_type": "LoraLoaderModelOnly", "inputs": {}})
        wf["lora"]["inputs"] = {"model": ["unet", 0], "lora_name": chosen, "strength_model": float(p["lora_strength"])}
        wf["guider"]["inputs"]["model"] = ["lora", 0]
    else:
        wf["guider"]["inputs"]["model"] = ["unet", 0]
    images = inp.get("images") or []
    if not images:
        raise ValueError("I2V requires input.images with one base64 image")
    # The template contains load_image with first_frame wired to it.
    load = next((v for v in wf.values() if isinstance(v, dict) and v.get("class_type") == "LoadImage"), None)
    if load is None:
        raise ValueError("workflow is missing LoadImage")
    name = _upload(images[0])
    load["inputs"]["image"] = name
    return wf

def _upload(item):
    import requests
    name = pathlib.Path(item.get("name", "input.png")).name
    raw = item.get("data") or item.get("base64")
    if not raw:
        raise ValueError("image item needs data/base64")
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    payload = base64.b64decode(raw)
    r = requests.post(f"{COMFY}/upload/image", files={"image": (name, payload, "application/octet-stream")},
                      data={"overwrite": "true", "type": "input"}, timeout=120)
    r.raise_for_status()
    return r.json()["name"]

def _run(inp):
    import requests
    p = _params(inp)
    job_id = inp.get("job_id", "unknown")
    try:
        from profiling import start_profiler
        profiler = start_profiler(job_id, {**p, "frames": _frames(p["duration"])})
        profiler.start()
    except Exception:
        profiler = None
    if profiler:
        profiler.begin("workflow_prepare", "🧩 准备 H3 workflow")
    wf = _workflow(inp, p)
    if profiler:
        profiler.end("workflow_prepare", "🧩 H3 workflow 准备完成")
    client = str(uuid.uuid4())
    if profiler:
        profiler.begin("comfyui_execution", "🎯 提交 ComfyUI，Sampling 详细 callback 当前不可用")
    q = requests.post(f"{COMFY}/prompt", json={"prompt": wf, "client_id": client}, timeout=120)
    q.raise_for_status()
    prompt_id = q.json()["prompt_id"]
    deadline = time.time() + int(os.getenv("JOB_TIMEOUT_SECONDS", "1800"))
    while time.time() < deadline:
        h = requests.get(f"{COMFY}/history/{prompt_id}", timeout=30).json()
        if prompt_id in h:
            item = h[prompt_id]
            if item.get("status", {}).get("status_str") == "error":
                raise RuntimeError(_json(item.get("status")))
            outputs = item.get("outputs", {})
            for out in outputs.values():
                for f in out.get("gifs", []) + out.get("videos", []) + out.get("images", []):
                    if f.get("filename", "").lower().endswith(".mp4"):
                        data = requests.get(f"{COMFY}/view", params={"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}, timeout=300).content
                        result = {"video": base64.b64encode(data).decode(), "filename": f["filename"],
                                  "model_profile": MODEL, "attention": os.getenv("ENABLE_ATTENTION", "0"),
                                  "steps": p["steps"], "warnings": (["negative_prompt and cfg are accepted for client compatibility but are not used by native H3"] if p["negative_prompt"] or p["cfg"] != 1.0 else [])}
                        if profiler:
                            profiler.end("comfyui_execution", "✅ ComfyUI 执行完成（包含采样及后处理，底层未提供分段 callback）")
                            profiler.finish(result)
                        return result
        time.sleep(2)
    raise TimeoutError(f"ComfyUI job timed out after {os.getenv('JOB_TIMEOUT_SECONDS', '1800')}s")

def handler(job):
    try:
        payload = dict(job.get("input") or {})
        payload.setdefault("job_id", job.get("id", "unknown"))
        return _run(payload)
    except Exception as exc:
        print(f"[h3-worker] job failed: {type(exc).__name__}: {exc}", flush=True)
        return {"error": str(exc), "type": type(exc).__name__}

if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
