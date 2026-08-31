from pathlib import Path


path = Path("/handler.py")
text = path.read_text()

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
