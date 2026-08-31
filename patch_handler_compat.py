from pathlib import Path


path = Path("/handler.py")
text = path.read_text()

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
        try:
            response = requests.get(f"http://{COMFY_HOST}/object_info", timeout=30)
            response.raise_for_status()
            node_list = list(response.json().keys())
        except Exception as exc:
            print(f"worker-comfyui - Could not read ComfyUI node list: {exc}")
            node_list = []
        return {"node_list": node_list}

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

path.write_text(text.replace(old, new, 1))
print("Patched /handler.py for RunOnRunpod compatibility")
