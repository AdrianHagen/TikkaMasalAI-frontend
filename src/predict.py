from typing import Any, Dict, Optional


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
