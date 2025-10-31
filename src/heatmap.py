import base64
import io
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


def _try_parse_base64_image(b64_str: str) -> Optional[Image.Image]:
    """Decode a base64 string (optionally a data URL) into a PIL Image.

    Handles whitespace, missing padding, and data URL prefixes like
    "data:image/png;base64,XXXXX".
    """
    if not isinstance(b64_str, str):
        return None
    try:
        s = b64_str.strip()
        # Strip data URL prefix if present
        if s.startswith("data:") and "," in s:
            s = s.split(",", 1)[1]
        # Remove whitespace
        s = "".join(s.split())
        # Fix padding if needed
        pad = (-len(s)) % 4
        if pad:
            s += "=" * pad
        raw = base64.b64decode(s, validate=False)
        return Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        return None


def find_heatmap_in_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Find a heatmap or overlay image in a flexible JSON payload.

    Returns a dict with one of:
    - {"overlay_image": PIL.Image}
    - {"heatmap": np.ndarray} with values in any real range
    - {"error": str}
    """
    if not isinstance(payload, dict):
        return {"error": "Invalid explanation payload (not a dict)."}

    # Common direct keys for overlay images (base64 or nested)
    for key in (
        "overlay",
        "overlay_image",
        "heatmap_image",
        "image",
        "image_base64",
        "overlay_base64",
        "base64",
        "b64",
        "png",
        "jpeg",
        "jpg",
        "webp",
    ):
        val = payload.get(key)
        if isinstance(val, str):
            img = _try_parse_base64_image(val)
            if img is not None:
                return {"overlay_image": img}

    # Common keys for array heatmap
    candidates = [
        payload.get("heatmap"),
        payload.get("saliency"),
        payload.get("attention"),
        payload.get("mask"),
    ]
    # Look in a nested "explanation" or similar
    for container_key in ("explanation", "result", "data"):
        sub = payload.get(container_key)
        if isinstance(sub, dict):
            candidates.extend(
                [
                    sub.get("heatmap"),
                    sub.get("saliency"),
                    sub.get("attention"),
                    sub.get("mask"),
                ]
            )
            for key in (
                "overlay",
                "overlay_image",
                "heatmap_image",
                "image",
                "image_base64",
                "overlay_base64",
                "base64",
                "b64",
                "png",
                "jpeg",
                "jpg",
                "webp",
            ):
                val = sub.get(key)
                if isinstance(val, str):
                    img = _try_parse_base64_image(val)
                    if img is not None:
                        return {"overlay_image": img}

    for c in candidates:
        if isinstance(c, list) and c and isinstance(c[0], (list, float, int)):
            try:
                arr = np.array(c, dtype=float)
                if arr.ndim == 2:
                    return {"heatmap": arr}
                if arr.ndim == 3 and arr.shape[2] == 1:
                    return {"heatmap": arr[..., 0]}
            except Exception:
                continue
        elif isinstance(c, str):
            # Could be base64 of a grayscale heatmap; attempt parse as image
            img = _try_parse_base64_image(c)
            if img is not None:
                # Convert to grayscale array as heatmap
                gray = img.convert("L")
                return {"heatmap": np.array(gray, dtype=float)}

    # Deep recursive search: try to find any base64 string that decodes to an image,
    # or any list-of-lists numeric array anywhere in the structure.
    def _search(obj: Any, depth: int = 0) -> Optional[Dict[str, Any]]:
        if depth > 4:
            return None
        if isinstance(obj, str):
            img = _try_parse_base64_image(obj)
            if img is not None:
                return {"overlay_image": img}
            return None
        if isinstance(obj, dict):
            # Favor image-like keys first
            for k, v in obj.items():
                if isinstance(v, str) and any(
                    t in k.lower()
                    for t in (
                        "image",
                        "overlay",
                        "heatmap",
                        "b64",
                        "base64",
                        "png",
                        "jpeg",
                        "jpg",
                        "webp",
                    )
                ):
                    img = _try_parse_base64_image(v)
                    if img is not None:
                        return {"overlay_image": img}
            # Then recurse
            for v in obj.values():
                found = _search(v, depth + 1)
                if found:
                    return found
            return None
        if isinstance(obj, list):
            # Could be a heatmap array or a list of nested structures
            if obj and isinstance(obj[0], (list, float, int)):
                try:
                    arr = np.array(obj, dtype=float)
                    if arr.ndim == 2:
                        return {"heatmap": arr}
                    if arr.ndim == 3 and arr.shape[2] == 1:
                        return {"heatmap": arr[..., 0]}
                except Exception:
                    pass
            for v in obj:
                found = _search(v, depth + 1)
                if found:
                    return found
            return None
        return None

    deep_found = _search(payload)
    if deep_found:
        return deep_found

    return {"error": "No heatmap or overlay image found in payload."}


def overlay_heatmap_on_image(
    base_img: Image.Image,
    heatmap: np.ndarray,
    opacity: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Overlay a heatmap onto an image using a red transparency mask.

    - base_img: PIL Image
    - heatmap: 2D numpy array
    - opacity: blending factor for heatmap contribution (0..1)
    - color: RGB color for the heatmap overlay
    """
    if not (0.0 <= opacity <= 1.0):
        opacity = max(0.0, min(1.0, opacity))

    base_rgba = base_img.convert("RGBA")
    w, h = base_rgba.size

    # Normalize heatmap to 0..1
    hm = np.array(heatmap, dtype=float)
    if hm.ndim != 2:
        # Try to squeeze if it has singleton dims
        hm = np.squeeze(hm)
        if hm.ndim != 2:
            raise ValueError("Heatmap must be 2D after squeeze.")

    # Resize to image size
    # Convert heatmap to an 8-bit grayscale image for resizing
    hm_min = np.nanmin(hm)
    hm_max = np.nanmax(hm)
    if not np.isfinite(hm_min) or not np.isfinite(hm_max) or hm_max - hm_min < 1e-12:
        # Degenerate map: use zeros
        hm_norm = np.zeros((h, w), dtype=np.float32)
    else:
        hm_norm = (hm - hm_min) / (hm_max - hm_min)
        hm_img = Image.fromarray((hm_norm * 255).astype(np.uint8), mode="L")
        hm_img = hm_img.resize((w, h), resample=Image.BILINEAR)
        hm_norm = np.asarray(hm_img, dtype=np.float32) / 255.0

    # Create colored overlay with per-pixel alpha = hm_norm * opacity
    r, g, b = color
    overlay = Image.new("RGBA", (w, h), (r, g, b, 0))
    alpha_layer = Image.fromarray((hm_norm * opacity * 255).astype(np.uint8), mode="L")
    overlay.putalpha(alpha_layer)

    blended = Image.alpha_composite(base_rgba, overlay)
    return blended.convert("RGB")
