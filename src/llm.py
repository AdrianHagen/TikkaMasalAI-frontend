from typing import Any, Dict, Optional


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
