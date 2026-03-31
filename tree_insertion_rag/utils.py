from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Iterable

from .models import TreeInsertionError


NODE_REQUIRED_FIELDS = {"node_name", "node_id", "node_type", "annotation"}


def ensure_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def has_required_node_fields(value: Any) -> bool:
    return isinstance(value, dict) and NODE_REQUIRED_FIELDS.issubset(value.keys())


def validate_node_payload(node: dict[str, Any], allow_children_missing: bool = False) -> None:
    if not isinstance(node, dict):
        raise TreeInsertionError("node must be a dictionary")

    missing = [field for field in NODE_REQUIRED_FIELDS if field not in node]
    if missing:
        raise TreeInsertionError(f"node is missing required fields: {', '.join(sorted(missing))}")

    if not allow_children_missing and node.get("node_type") == "parent":
        children = node.get("children")
        if not isinstance(children, list):
            raise TreeInsertionError("parent node must contain a list field named 'children'")


def normalize_text(text: str) -> str:
    raw = ensure_str(text).lower()
    raw = raw.replace("_", " ")
    raw = re.sub(r"[\r\n\t]+", " ", raw)
    raw = re.sub(r"[^\w\u4e00-\u9fff]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    ascii_tokens = re.findall(r"[a-z0-9]+", normalized)
    cjk_segments = re.findall(r"[\u4e00-\u9fff]+", normalized)
    cjk_tokens: list[str] = []
    for segment in cjk_segments:
        cjk_tokens.extend(list(segment))
        if len(segment) > 1:
            cjk_tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
    return ascii_tokens + cjk_tokens


def build_node_text(node_name: str, annotation: str) -> str:
    name = ensure_str(node_name)
    note = ensure_str(annotation)
    if name and note:
        return f"{name} {note}"
    return name or note


def summarize_text(text: str, limit: int = 80) -> str:
    text = ensure_str(text)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def counter_cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(left[token] * right.get(token, 0) for token in left)
    if numerator <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def lexical_similarity(left: str, right: str) -> float:
    return counter_cosine_similarity(Counter(tokenize(left)), Counter(tokenize(right)))


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_list = list(left)
    right_list = list(right)
    if len(left_list) != len(right_list) or not left_list:
        return 0.0
    numerator = sum(a * b for a, b in zip(left_list, right_list))
    if numerator <= 0:
        return 0.0
    left_norm = math.sqrt(sum(a * a for a in left_list))
    right_norm = math.sqrt(sum(b * b for b in right_list))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def average_top_scores(scores: list[float], topn: int = 2) -> float:
    if not scores:
        return 0.0
    selected = sorted(scores, reverse=True)[: max(1, topn)]
    return sum(selected) / len(selected)


def min_max_normalize(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}

    values = list(score_map.values())
    highest = max(values)
    lowest = min(values)
    if math.isclose(highest, lowest):
        if highest <= 0:
            return {key: 0.0 for key in score_map}
        return {key: 1.0 for key in score_map}

    return {key: (value - lowest) / (highest - lowest) for key, value in score_map.items()}
