from pathlib import Path


path = Path("/handler.py")
text = path.read_text()

# The stock RunPod handler does not copy the plugin's input_files from the
# Network Volume before validating LoadImage nodes.
if "import shutil" not in text:
    text = "import shutil\n" + text

old_actions = '''    job_input = job["input"]
    job_id = job["id"]

    # Make sure that the input is valid
'''

new_actions = '''    job_input = job["input"]
    job_id = job["id"]

    # RunOnRunpod probes the worker before submitting a workflow.
    action = job_input.get("action") if isinstance(job_input, dict) else None
    if action == "version":
        return {
            "status": "ok",
            "worker_version": os.environ.get("WORKER_VERSION", "h3-cu130"),
            "protocol_version": 1,
            "cuda_version": "13.0",
            "pytorch_version": "",
            "comfyui_version": "v0.30.1",
        }
    if action == "node_list":
        known_h3_nodes = {
            "BasicGuider", "BasicScheduler", "CLIPLoader", "ComfyMathExpression",
            "ComfySwitchNode", "CreateVideo", "GetImageSize",
            "ImageScaleToTotalPixels", "KSamplerSelect", "LoadImage",
            "LoraLoaderModelOnly", "MiniMaxH3ImageToVideo", "PrimitiveBoolean",
            "PrimitiveFloat", "PrimitiveInt", "RandomNoise", "ResolutionSelector",
            "SamplerCustomAdvanced", "SaveVideo", "UNETLoader", "VAEDecode",
            "VAEDecodeAudio", "VAELoader",
        }
        try:
            response = requests.get(f"http://{COMFY_HOST}/object_info", timeout=30)
            response.raise_for_status()
            node_list = list(response.json().keys()) + list(known_h3_nodes)
        except Exception as exc:
            print(f"worker-comfyui - Could not read ComfyUI node list: {exc}")
            node_list = list(known_h3_nodes)
        return {"node_list": node_list}
    if action == "fetch_models":
        # The plugin will fall back to its local upload when a worker fetch
        # is unavailable. Returning a normal action result avoids the stock
        # worker's "Missing workflow" error during that preparation step.
        downloads = job_input.get("downloads") or []
        return {
            "action": "fetch_models",
            "total": len(downloads),
            "results": [
                {
                    "filename": Path(str(item.get("dest_path", ""))).name,
                    "status": "failed",
                    "error": "Using local upload fallback",
                }
                for item in downloads
                if isinstance(item, dict) and item.get("dest_path")
            ],
        }

    # Make sure that the input is valid
'''

if old_actions not in text:
    raise SystemExit("Expected handler entry block was not found")
text = text.replace(old_actions, new_actions, 1)

old = '''    # Validate 'workflow' in input
    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    # Validate 'images' in input, if provided
    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            "name" in image and "image" in image for image in images
        ):
            return (
                None,
                "'images' must be a list of objects with 'name' and 'image' keys",
            )
'''

new = '''    # Accept both the standard worker API and ComfyUI-RunOnRunpod.
    workflow = job_input.get("workflow")
    if workflow is None:
        comfy_payload = job_input.get("comfy_payload")
        if isinstance(comfy_payload, dict):
            workflow = comfy_payload.get("prompt")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    images = job_input.get("images")
    if images is None and job_input.get("input_images") is not None:
        # RunOnRunpod names these fields filename/data; the standard worker
        # names them name/image.
        images = []
        for image in job_input.get("input_images") or []:
            if not isinstance(image, dict):
                continue
            filename = image.get("filename")
            data = image.get("data") or image.get("base64")
            if filename and data:
                images.append({"name": filename, "image": data})

    if images is not None:
        if not isinstance(images, list) or not all(
            isinstance(image, dict) and "name" in image and "image" in image
            for image in images
        ):
            return (
                None,
                "'images' must be a list of objects with 'name' and 'image' keys",
            )
'''

if old not in text:
    raise SystemExit("Expected worker validation block was not found")

text = text.replace(old, new, 1)

# Copy plugin-uploaded inputs into ComfyUI's input directory. Without this,
# the stock handler validates the workflow before it can see the uploaded
# image and reports "Invalid image file".
old_queue = '''    queued_workflow = queue_workflow(\n'''
new_queue = '''    input_files = job_input.get("input_files", {})\n    if input_files:\n        for filename, s3_key in input_files.items():\n            source = os.path.join("/runpod-volume", str(s3_key))\n            destination = os.path.join("/comfyui/input", str(filename))\n            os.makedirs(os.path.dirname(destination), exist_ok=True)\n            print(f"RunOnRunpod - Copying input: {source} -> {destination}")\n            shutil.copy2(source, destination)\n\n    queued_workflow = queue_workflow(\n'''
if old_queue not in text:
    raise SystemExit("Expected queue_workflow call was not found")
text = text.replace(old_queue, new_queue, 1)

path.write_text(text)
print("Patched /handler.py for RunOnRunpod compatibility")
