"""RunPod handler that adapts the ComfyUI HTTP API to Serverless jobs."""
import base64, copy, json, os, pathlib, time, uuid
import shutil
from typing import Any


COMFY = f"http://127.0.0.1:{os.getenv('COMFYUI_PORT', '8188')}"
TEMPLATE = pathlib.Path(os.getenv('WORKFLOW_TEMPLATE', str(pathlib.Path(__file__).parent / 'workflows/h3_i2v_api.json')))
DEFAULTS = {"mode": "i2v", "width": 1344, "height": 768, "duration": 15, "fps": 24, "steps": 8,
            "seed": 42, "sampler": "res_multistep", "scheduler": "simple",
            "cfg": 1.0, "lora_strength": 1.0, "prompt": "", "negative_prompt": ""}
MODE_ALIASES = {
    "image_to_video": "i2v",
    "video_to_video": "v2v",
    "image_video_mix": "rv2v",
    "audio_reference": "r2v",
}
MODEL = os.getenv("MODEL_PROFILE", "blackwell_fp8")
LORAS = {4: os.getenv("LORA_4STEP", "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors"),
         8: os.getenv("LORA_8STEP", "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors")}

def _json(data):
    return json.dumps(data, ensure_ascii=False)

def _params(inp):
    p = {**DEFAULTS, **(inp.get("params") or {})}
    p["mode"] = str(p["mode"]).lower()
    p["mode"] = MODE_ALIASES.get(p["mode"], p["mode"])
    if p["mode"] not in ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"):
        raise ValueError("mode must be one of t2v, i2v, fl2v, r2v, v2v, rv2v")
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
    if p["mode"] != "i2v" and not inp.get("workflow"):
        raise ValueError(f"mode={p['mode']} requires a matching ComfyUI Desktop API workflow")
    supplied = inp.get("workflow")
    if isinstance(supplied, dict) and isinstance(supplied.get("nodes"), list):
        raise ValueError(
            "workflow is a ComfyUI canvas export; open it in ComfyUI Desktop and "
            "choose Save (API Format) before sending it to RunPod"
        )
    wf = copy.deepcopy(inp.get("workflow") or json.loads(TEMPLATE.read_text()))
    # Keep node ids stable in the shipped API template; exported Desktop workflows can
    # also be used when they retain these semantic node types and fields.
    typed = {}
    for k, v in wf.items():
        if isinstance(v, dict) and v.get("class_type"):
            typed.setdefault(v["class_type"], []).append(v)

    def node(kind):
        if kind not in typed:
            raise ValueError(f"workflow is missing required node: {kind}")
        return typed[kind][0]["inputs"]

    def first_node(*kinds):
        for kind in kinds:
            if typed.get(kind):
                return typed[kind][0]
        return None

    # A supplied ComfyUI Desktop API workflow is authoritative.  The worker only
    # fills values that have a matching native node, so Director/R2V/V2V graphs
    # can be sent without forcing them through the I2V template below.
    cond_node = first_node("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo")
    if cond_node is None:
        if not inp.get("workflow"):
            raise ValueError("workflow is missing MiniMax H3 I2V/ReferenceToVideo node")
        # Custom nodes such as MiniMax H3 Easy/Director own their conditioning
        # schema.  A Desktop API workflow is submitted unchanged; only matching
        # loader assets below are rebound.
    else:
        cond_inputs = cond_node["inputs"]
        cond_inputs.update({"prompt": p["prompt"], "width": p["width"], "height": p["height"], "length": _frames(p["duration"])})
    if typed.get("RandomNoise"):
        node("RandomNoise")["noise_seed"] = int(p["seed"])
    if typed.get("KSamplerSelect"):
        node("KSamplerSelect")["sampler_name"] = p["sampler"]
    if typed.get("BasicScheduler"):
        node("BasicScheduler").update({"steps": int(p["steps"]), "scheduler": p["scheduler"]})
    if typed.get("CreateVideo"):
        node("CreateVideo")["fps"] = int(p["fps"])
    # 6 steps has no separately trained official FL2V LoRA; 8-step is the closest
    # supported Turbo operating point.  20 is the base/reference benchmark.
    lora_nodes = []
    for _, v in wf.items():
        if isinstance(v, dict) and v.get("class_type") in ("LoraLoaderModelOnly", "LoraLoader"):
            lora_nodes.append(v)
            v["inputs"]["lora_name"] = LORAS.get(int(p["steps"]), "")
            v["inputs"]["strength_model"] = float(p["lora_strength"])
    if int(p["steps"]) in (4, 6, 8) and "guider" in wf and "unet" in wf:
        chosen = LORAS[4] if int(p["steps"]) == 4 else LORAS[8]
        wf.setdefault("lora", {"class_type": "LoraLoaderModelOnly", "inputs": {}})
        wf["lora"]["inputs"] = {"model": ["unet", 0], "lora_name": chosen, "strength_model": float(p["lora_strength"])}
        wf["guider"]["inputs"]["model"] = ["lora", 0]
    elif "guider" in wf and "unet" in wf:
        wf["guider"]["inputs"]["model"] = ["unet", 0]
    _bind_assets(wf, inp)
    return wf

def _asset_groups(inp):
    """Accept both the old `images` field and the unified asset contract."""
    assets = inp.get("assets") or []
    groups = {"image": list(inp.get("images") or []), "video": [], "audio": []}
    for asset in assets:
        kind = str(asset.get("type", "image")).lower()
        if kind in ("image", "video", "audio"):
            groups[kind].append(asset)
    refs = inp.get("references") or {}
    for kind in groups:
        for asset in refs.get(f"{kind}s", []) or []:
            groups[kind].append(asset)
    return groups

def _bind_assets(wf, inp):
    groups = _asset_groups(inp)
    uploaded = {kind: [_upload(x) for x in items] for kind, items in groups.items()}
    typed = {}
    for value in wf.values():
        if isinstance(value, dict) and value.get("class_type"):
            typed.setdefault(value["class_type"], []).append(value)

    image_nodes = typed.get("LoadImage", [])
    video_nodes = typed.get("LoadVideo", []) + typed.get("VHS_LoadVideo", [])
    audio_nodes = typed.get("LoadAudio", []) + typed.get("VHS_LoadAudio", [])
    # Native H3 graphs use LoadImage/LoadVideo/LoadAudio or their VHS variants.
    # Bind in graph order; an explicit `slot` can target a particular node.
    for index, value in enumerate(uploaded["image"]):
        if index < len(image_nodes):
            image_nodes[index]["inputs"]["image"] = value
    for index, value in enumerate(uploaded["video"]):
        if index < len(video_nodes):
            video_nodes[index]["inputs"]["video"] = value
    for index, value in enumerate(uploaded["audio"]):
        if index < len(audio_nodes):
            audio_nodes[index]["inputs"]["audio"] = value

    for kind, nodes in (("image", image_nodes), ("video", video_nodes), ("audio", audio_nodes)):
        if len(uploaded[kind]) > len(nodes):
            raise ValueError(
                f"received {len(uploaded[kind])} {kind} assets but workflow has only "
                f"{len(nodes)} matching loader nodes; add loaders in ComfyUI Desktop"
            )

    mode = str((inp.get("params") or {}).get("mode", DEFAULTS["mode"])).lower()
    required = {"i2v": (1, 0, 0), "fl2v": (1, 0, 0), "r2v": (0, 0, 0),
                "v2v": (0, 1, 0), "rv2v": (0, 1, 0), "t2v": (0, 0, 0)}[mode]
    if len(uploaded["image"]) < required[0] and mode in ("i2v", "fl2v"):
        raise ValueError(f"{mode} requires at least one image asset")
    if len(uploaded["video"]) < required[1] and mode in ("v2v", "rv2v"):
        raise ValueError(f"{mode} requires at least one video asset")
    if mode == "r2v" and not any(uploaded.values()):
        raise ValueError("r2v requires at least one reference image, video, or audio asset")

def _upload(item, asset_type="image"):
    import requests
    name = pathlib.Path(item.get("name", "input.png")).name
    raw = item.get("data") or item.get("base64")
    if not raw:
        raise ValueError("image item needs data/base64")
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    payload = base64.b64decode(raw)
    # ComfyUI's upload endpoint accepts all input media as multipart content;
    # the loader node determines whether it is treated as image/video/audio.
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
                                  "mode": p["mode"],
                                  "steps": p["steps"], "warnings": (["negative_prompt and cfg are accepted for client compatibility but are not used by native H3"] if p["negative_prompt"] or p["cfg"] != 1.0 else [])}
                        if profiler:
                            profiler.end("comfyui_execution", "✅ ComfyUI 执行完成（包含采样及后处理，底层未提供分段 callback）")
                            profiler.finish(result)
                        return result
        time.sleep(2)
    raise TimeoutError(f"ComfyUI job timed out after {os.getenv('JOB_TIMEOUT_SECONDS', '1800')}s")


def _wait_comfy(timeout=300):
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{COMFY}/object_info", timeout=5).ok:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _run_runonrunpod(job):
    """Compatibility path for the installed ComfyUI-RunOnRunpod plugin.

    The normal H3 API contract above remains unchanged.  This path accepts the
    plugin's input_files/workflow contract and persists outputs where its S3
    client expects them.
    """
    import requests
    inp = dict(job.get("input") or {})
    action = inp.get("action")
    if action == "version":
        ready = _wait_comfy()
        try:
            import torch
            torch_version, cuda = torch.__version__, torch.version.cuda or ""
        except Exception:
            torch_version, cuda = "", ""
        return {"status": "ok" if ready else "comfy_not_ready", "worker_version": "h3",
                "protocol_version": 1, "cuda_version": cuda, "pytorch_version": torch_version,
                "comfyui_version": os.getenv("COMFYUI_VERSION", "0.32.0")}
    if action == "node_list":
        return {"node_list": list(requests.get(f"{COMFY}/object_info", timeout=120).json())}
    if action == "fetch_models":
        # The Feature Easy startup already provisions its known H3 files.  Any
        # unknown file is deliberately reported for the plugin's local-upload
        # fallback, avoiding a second downloader in the production handler.
        return {"action": "fetch_models", "total": len(inp.get("downloads") or []),
                "results": [{"filename": os.path.basename(x.get("dest_path", "")),
                             "status": "failed", "error": "use plugin local-upload fallback"}
                            for x in (inp.get("downloads") or [])]}

    workflow = inp.get("workflow")
    if not isinstance(workflow, dict):
        raise ValueError("RunOnRunpod workflow is missing")
    for filename, s3_key in (inp.get("input_files") or {}).items():
        src = os.path.join(os.getenv("MODEL_VOLUME_PATH", "/runpod-volume"), s3_key)
        dest = os.path.join("/comfyui/input", pathlib.Path(filename).name)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"input file missing on Network Volume: {s3_key}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
    q = requests.post(f"{COMFY}/prompt", json={"prompt": workflow}, timeout=120)
    q.raise_for_status()
    prompt_id = q.json()["prompt_id"]
    deadline = time.time() + int(os.getenv("JOB_TIMEOUT_SECONDS", "1800"))
    while time.time() < deadline:
        history = requests.get(f"{COMFY}/history/{prompt_id}", timeout=30).json()
        if prompt_id in history:
            item = history[prompt_id]
            if item.get("status", {}).get("status_str") == "error":
                return {"error": _json(item.get("status"))}
            if item.get("status", {}).get("completed") or item.get("status", {}).get("status_str") == "success":
                root = os.getenv("MODEL_VOLUME_PATH", "/runpod-volume")
                job_dir = time.strftime("%Y%m%d%H%M%S") + "_" + str(job.get("id", "unknown"))
                output_files = []
                for out in item.get("outputs", {}).values():
                    for key in ("gifs", "videos", "images", "audio"):
                        for f in out.get(key, []):
                            src = os.path.join("/comfyui/output", f.get("subfolder", ""), f["filename"])
                            if os.path.isfile(src):
                                rel = os.path.join(job_dir, pathlib.Path(f["filename"]).name)
                                dest = os.path.join(root, "outputs", rel)
                                os.makedirs(os.path.dirname(dest), exist_ok=True)
                                shutil.copy2(src, dest)
                                output_files.append(rel)
                return {"status": "success", "output_count": len(output_files), "output_files": output_files}
        time.sleep(1)
    raise TimeoutError("RunOnRunpod workflow timed out")

def handler(job):
    try:
        if (job.get("input") or {}).get("action") in ("version", "node_list", "fetch_models"):
            return _run_runonrunpod(job)
        if (job.get("input") or {}).get("input_files") and (job.get("input") or {}).get("workflow"):
            return _run_runonrunpod(job)
        payload = dict(job.get("input") or {})
        payload.setdefault("job_id", job.get("id", "unknown"))
        return _run(payload)
    except Exception as exc:
        print(f"[h3-worker] job failed: {type(exc).__name__}: {exc}", flush=True)
        return {"error": str(exc), "type": type(exc).__name__}

if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
