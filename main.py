"""Streamlit entrypoint for the TikkaMasalAI dish classifier frontend.

Refactored layout:
- Sidebar hosts the uploader and app title.
- Main area shows the uploaded image, the predicted dish, and tabs for:
    1) Get Recipe  2) Explain Prediction  3) Raw JSON

This reduces scrolling and groups controls vs. results more clearly.
"""

import io
import base64
import mimetypes
from typing import Any, Dict, Optional

import requests
import streamlit as st
from PIL import Image
import numpy as np
import hashlib

DEFAULT_API_URL = "http://127.0.0.1:8000/predict"
DEFAULT_LLM_URL = "http://127.0.0.1:8000/llm/generate"
DEFAULT_EXPLAIN_URL = "http://127.0.0.1:8000/predict/explain"


def get_api_url() -> str:
    """Allow overriding the API URL via Streamlit secrets or query params."""
    # Prioritize secrets so deployments can configure without code changes.
    try:
        api_url = st.secrets.get("api_url", DEFAULT_API_URL)
    except Exception:
        api_url = DEFAULT_API_URL
    # Allow quick overrides via query param for local testing.
    api_override = st.query_params.get("api_url")
    if isinstance(api_override, list) and api_override:
        api_url = api_override[-1]
    elif isinstance(api_override, str) and api_override:
        api_url = api_override
    return api_url


def get_llm_url() -> str:
    """Allow overriding the LLM URL via Streamlit secrets or query params."""
    try:
        llm_url = st.secrets.get("llm_url", DEFAULT_LLM_URL)
    except Exception:
        llm_url = DEFAULT_LLM_URL
    llm_override = st.query_params.get("llm_url")
    if isinstance(llm_override, list) and llm_override:
        llm_url = llm_override[-1]
    elif isinstance(llm_override, str) and llm_override:
        llm_url = llm_override
    return llm_url


def get_explain_url() -> str:
    """Allow overriding the explain URL via Streamlit secrets or query params."""
    try:
        explain_url = st.secrets.get("explain_url", DEFAULT_EXPLAIN_URL)
    except Exception:
        explain_url = DEFAULT_EXPLAIN_URL
    explain_override = st.query_params.get("explain_url")
    if isinstance(explain_override, list) and explain_override:
        explain_url = explain_override[-1]
    elif isinstance(explain_override, str) and explain_override:
        explain_url = explain_override
    return explain_url


def extract_primary_label(payload: Dict[str, Any]) -> Optional[str]:
    """Return the most likely label key from the prediction payload."""
    for key in (
        "dish",
        "prediction",
        "predictions",
        "label",
        "class",
        "result",
        "scores",
    ):
        if key in payload:
            value = payload[key]
            # Handle nested responses such as {"prediction": {"label": "..."}}
            if isinstance(value, dict):
                nested_label = extract_primary_label(value)
                if nested_label:
                    return nested_label
                # Support score dictionaries like {"predictions": {"foo": 0.9}}
                numeric_items = {
                    k: v for k, v in value.items() if isinstance(v, (int, float))
                }
                if numeric_items:
                    return max(numeric_items.items(), key=lambda item: item[1])[0]
            elif isinstance(value, (str, int, float)):
                return str(value)
    return None


def extract_llm_text(payload: Dict[str, Any]) -> Optional[str]:
    """Try to extract a human-readable text field from various LLM response shapes.

    Supports common patterns like:
    - {"text": "..."}
    - {"response": "..."}
    - {"output": "..."}
    - {"choices": [{"message": {"content": "..."}}]}
    - {"choices": [{"text": "..."}]}
    - {"message": {"content": "..."}}
    - {"data": {"text": "..."}} or similar nested dicts.
    """
    if not isinstance(payload, dict):
        return None

    # Direct text-like keys
    for key in ("text", "response", "output", "content", "result"):
        if isinstance(payload.get(key), str):
            return payload[key]

    # OpenAI-like choices array
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            # Chat style
            msg = first.get("message") or first.get("delta")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
            # Text completion style
            text_val = first.get("text")
            if isinstance(text_val, str) and text_val.strip():
                return text_val

    # Nested dict fallbacks (one level deep)
    for v in payload.values():
        if isinstance(v, dict):
            nested = extract_llm_text(v)
            if nested:
                return nested

    return None


def extract_top_scores(
    payload: Dict[str, Any], top_n: int = 5
) -> list[tuple[str, float]]:
    """Extract top-N (label, score) pairs from a flexible prediction payload.

    Tries common patterns:
    - Dict[str, number] under keys like 'predictions', 'scores', 'probabilities', 'probs'.
    - A list of {label: ..., score/probability: ...} dicts.
    - Direct dict[str, number] at the root or nested levels.
    """
    if not isinstance(payload, dict):
        return []

    def parse_score_dict(obj: Any) -> Optional[dict[str, float]]:
        if isinstance(obj, dict) and obj:
            numeric = {
                k: float(v)
                for k, v in obj.items()
                if isinstance(k, str) and isinstance(v, (int, float))
            }
            if numeric:
                return numeric
        return None

    def parse_score_list(obj: Any) -> Optional[dict[str, float]]:
        if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            out: dict[str, float] = {}
            for item in obj:
                label = item.get("label") or item.get("class") or item.get("dish")
                score = (
                    item.get("score")
                    if isinstance(item.get("score"), (int, float))
                    else item.get("probability")
                )
                if isinstance(label, str) and isinstance(score, (int, float)):
                    out[label] = float(score)
            if out:
                return out
        return None

    candidates: list[dict[str, float]] = []

    def collect(obj: Any, depth: int = 0):
        if depth > 3:
            return
        if isinstance(obj, dict):
            # Prefer common keys first
            for key in ("predictions", "scores", "probabilities", "probs"):
                if key in obj:
                    v = obj[key]
                    m = parse_score_dict(v)
                    if m:
                        candidates.append(m)
                    else:
                        m2 = parse_score_list(v)
                        if m2:
                            candidates.append(m2)

            # Direct dict of scores
            m = parse_score_dict(obj)
            if m:
                candidates.append(m)

            for v in obj.values():
                collect(v, depth + 1)
        elif isinstance(obj, list):
            m = parse_score_list(obj)
            if m:
                candidates.append(m)
            for v in obj:
                collect(v, depth + 1)

    collect(payload)

    if not candidates:
        return []
    # Choose the largest candidate by number of classes
    best = max(candidates, key=lambda d: len(d))
    top = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_n)]
    return top


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


def main() -> None:
    st.set_page_config(page_title="TikkaMasalAI", page_icon=":curry:", layout="wide")

    # URLs (configurable via secrets/query)
    api_url = get_api_url()
    explain_url = get_explain_url()
    llm_url = get_llm_url()

    # Sidebar: App title and uploader
    st.sidebar.title("Tikka MasalAI 🍲")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a food photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, PNG, WEBP.",
    )

    # Gate main content on upload
    if not uploaded_file:
        st.title("TikkaMasalAI Dish Classifier")
        st.info("Please upload a food photo to get started.")
        return

    # Load image + compute a content hash to cache prediction across reruns
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    mime_type = (
        uploaded_file.type
        or mimetypes.guess_type(uploaded_file.name)[0]
        or "application/octet-stream"
    )
    # Sidebar preview of the uploaded image
    st.sidebar.image(image, caption="Preview", use_container_width=True)
    img_hash = hashlib.md5(image_bytes).hexdigest()

    # Cache prediction for same image across reruns
    cache_hit = (
        st.session_state.get("tikka_img_hash") == img_hash
        and st.session_state.get("tikka_prediction") is not None
    )
    if not cache_hit:
        with st.spinner("Identifying the dish..."):
            try:
                response = requests.post(
                    api_url,
                    files={"image": (uploaded_file.name, image_bytes, mime_type)},
                    timeout=30,
                )
                response.raise_for_status()
                pred_json = response.json()
            except requests.exceptions.RequestException as request_error:
                st.error(f"Request failed: {request_error}")
                return
            except ValueError:
                st.error("The service returned a non-JSON response.")
                return

        st.session_state["tikka_img_hash"] = img_hash
        st.session_state["tikka_prediction"] = pred_json
        primary = extract_primary_label(pred_json)
        st.session_state["tikka_primary_label"] = primary
        # Visual mark: toast to indicate prediction completed (once per new image)
        try:
            if primary:
                st.toast(f"Predicted: {primary.capitalize()}", icon="✅")
            else:
                st.toast("Prediction ready", icon="✅")
        except Exception:
            # Fallback for Streamlit versions without st.toast
            if primary:
                st.success(f"Predicted: {primary.capitalize()}")
        # Reset tab-specific session artifacts on new image
        st.session_state.pop("tikka_llm_response", None)
        st.session_state.pop("tikka_explain_raw", None)
    else:
        pred_json = st.session_state["tikka_prediction"]
        # Primary label may have been computed already
        if "tikka_primary_label" not in st.session_state:
            st.session_state["tikka_primary_label"] = extract_primary_label(pred_json)

    primary_label: Optional[str] = st.session_state.get("tikka_primary_label")

    # Main area: uploaded image first (centered and modest width), then the main prediction
    left, center, right = st.columns([1, 2, 1])
    if primary_label:
        st.markdown(
            f"## ✅ Predicted Dish: <span style='color:#16a34a'>{primary_label.capitalize()}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.header("Predicted Dish")
        st.info("Prediction received, but no obvious label was found in the response.")

    # Tabs for next steps
    tab1, tab2, tab3 = st.tabs(["Get Recipe", "Explain Prediction", "Raw JSON"])

    # Tab 1: Get Recipe (LLM)
    with tab1:
        st.subheader("Recipe helper")
        default_question = (
            f"How do I prepare {primary_label}? Provide ingredients and clear step-by-step cooking instructions."
            if primary_label
            else "How do I prepare this dish? Provide ingredients and clear step-by-step cooking instructions."
        )
        user_question = st.text_area(
            "Your recipe question:",
            value=default_question,
            height=120,
            help="You can edit the question before asking the LLM.",
        )
        ask = st.button("Get recipe from LLM", type="primary")

        if ask and user_question.strip():
            with st.spinner("Asking the LLM for a recipe..."):
                llm_error: Optional[str] = None
                llm_payload_candidates = [
                    {"prompt": user_question},
                    {"text": user_question},
                    {"input": user_question},
                    {"messages": [{"role": "user", "content": user_question}]},
                ]

                llm_response_json: Optional[Dict[str, Any]] = None
                for candidate in llm_payload_candidates:
                    try:
                        r = requests.post(llm_url, json=candidate, timeout=60)
                        r.raise_for_status()
                        llm_response_json = r.json()
                        break
                    except requests.exceptions.RequestException as e:
                        llm_error = str(e)
                    except ValueError:
                        llm_error = "The LLM service returned a non-JSON response."

                if llm_response_json is None:
                    st.error(
                        "Failed to get a response from the LLM service. "
                        f"Last error: {llm_error or 'Unknown error'}"
                    )
                else:
                    st.session_state["tikka_llm_response"] = llm_response_json
                    llm_text = extract_llm_text(llm_response_json)
                    if llm_text:
                        st.subheader("Suggested recipe")
                        st.markdown(llm_text)
                    else:
                        st.info(
                            "Received an LLM response, but couldn't find any text to display."
                        )

        # If a previous response exists (e.g., after tab switch), show it
        if st.session_state.get("tikka_llm_response") and not ask:
            llm_text = extract_llm_text(st.session_state["tikka_llm_response"])
            if llm_text:
                st.subheader("Suggested recipe")
                st.markdown(llm_text)

    # Tab 2: Explain Prediction (Heatmap)
    with tab2:
        st.subheader("Explain prediction")
        # Show top-5 class probabilities from the earlier prediction
        top5 = extract_top_scores(pred_json, top_n=5)
        if top5:
            st.markdown("**Model certainty for top 5 predictions**")
            st.caption("Bars represent the model's certainty for each class.")
            for label, score in top5:
                # Convert to percentage if needed and clamp to [0, 100]
                pct = float(score)
                pct = pct * 100.0 if pct <= 1.0 else pct
                pct = max(0.0, min(100.0, pct))
                lcol, pcol, vcol = st.columns([2, 6, 2])
                with lcol:
                    st.markdown(f"**{label}**")
                with pcol:
                    st.progress(int(round(pct)))
                with vcol:
                    st.markdown(f"{pct:.1f}%")

        heatmap_opacity = st.slider(
            "Heatmap opacity", 0.1, 0.9, 0.5, 0.05, key="opacity_slider"
        )
        do_explain = st.button("Generate Heatmap")

        if do_explain:
            with st.spinner("Requesting explanation heatmap..."):
                try:
                    form_data = (
                        {"label": (None, primary_label)} if primary_label else None
                    )
                    response = requests.post(
                        explain_url,
                        files={"image": (uploaded_file.name, image_bytes, mime_type)},
                        data=form_data,
                        timeout=60,
                    )
                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "")
                    overlay_img: Optional[Image.Image] = None
                    heatmap_arr: Optional[np.ndarray] = None
                    raw_payload: Optional[Dict[str, Any]] = None

                    if content_type.startswith("image/"):
                        overlay_img = Image.open(io.BytesIO(response.content))
                    else:
                        try:
                            parsed = response.json()
                            raw_payload = parsed
                        except ValueError:
                            parsed = response.text.strip()

                        if isinstance(parsed, str):
                            maybe_img = _try_parse_base64_image(parsed)
                            if maybe_img is not None:
                                overlay_img = maybe_img
                            else:
                                st.info(
                                    "Explain service returned a string, but it wasn't a valid base64 image."
                                )
                        elif isinstance(parsed, dict):
                            found = find_heatmap_in_payload(parsed)
                            raw_payload = parsed
                            if "overlay_image" in found:
                                overlay_img = found["overlay_image"]
                            elif "heatmap" in found:
                                heatmap_arr = found["heatmap"]
                            else:
                                st.info(
                                    found.get("error", "No heatmap found in response.")
                                )
                        else:
                            st.info(
                                "Explain service returned an unsupported payload format."
                            )

                    # Persist raw payload for the Raw JSON tab
                    if raw_payload is not None:
                        st.session_state["tikka_explain_raw"] = raw_payload

                    if overlay_img is not None:
                        st.image(
                            overlay_img,
                            caption="Explanation overlay (from server)",
                            width=480,
                        )
                    elif heatmap_arr is not None:
                        try:
                            blended = overlay_heatmap_on_image(
                                image, heatmap_arr, opacity=heatmap_opacity
                            )
                            st.image(
                                blended,
                                caption="Heatmap overlay",
                                width=480,
                            )
                        except Exception as e:
                            st.error(f"Failed to overlay heatmap: {e}")
                except requests.exceptions.RequestException as request_error:
                    st.error(f"Explain request failed: {request_error}")

    # Tab 3: Raw JSON
    with tab3:
        st.subheader("Raw responses")
        with st.expander("Show Raw Classification Response"):
            st.json(pred_json)

        if st.session_state.get("tikka_llm_response") is not None:
            with st.expander("Show Raw LLM Response"):
                st.json(st.session_state["tikka_llm_response"])

        if st.session_state.get("tikka_explain_raw") is not None:
            with st.expander("Show Raw Explanation Response"):
                st.json(st.session_state["tikka_explain_raw"])


if __name__ == "__main__":
    main()
