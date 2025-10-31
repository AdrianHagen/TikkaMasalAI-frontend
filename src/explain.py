from typing import Any, Dict, Optional


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
