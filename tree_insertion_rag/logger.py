from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger(__name__)


def log_stage(stage: str, elapsed_ms: float, output: Any) -> None:
    logger.info(
        "tree_insertion stage=%s elapsed_ms=%.3f output=%s",
        stage,
        elapsed_ms,
        _serialize_output(output),
    )


def summarize_parsed_nodes(parsed_nodes: list[Any], limit: int = 5) -> dict[str, Any]:
    return {
        "count": len(parsed_nodes),
        "items": [
            {
                "node_id": node.node_id,
                "node_name": node.node_name,
                "node_type": node.node_type,
                "jsonpath": node.jsonpath,
                "depth": node.depth,
            }
            for node in parsed_nodes[:limit]
        ],
    }


def summarize_ranked_candidates(ranked_candidates: list[Any], limit: int = 5) -> dict[str, Any]:
    return {
        "count": len(ranked_candidates),
        "items": [
            {
                "node_id": ranked.candidate.node_id,
                "node_name": ranked.candidate.node_name,
                "node_type": ranked.candidate.node_type,
                "jsonpath": ranked.candidate.jsonpath,
                "final_score": round(ranked.final_score, 6),
                "semantic_score": round(ranked.semantic_score, 6),
                "sibling_score": round(ranked.sibling_score, 6),
                "path_prior_score": round(ranked.path_prior_score, 6),
            }
            for ranked in ranked_candidates[:limit]
        ],
    }


def summarize_selection(selected: Any, fallback: bool) -> dict[str, Any]:
    if selected is None:
        return {"selected": None, "used_fallback": fallback}
    return {
        "selected": {
            "node_id": selected.candidate.node_id,
            "node_name": selected.candidate.node_name,
            "node_type": selected.candidate.node_type,
            "jsonpath": selected.candidate.jsonpath,
            "final_score": round(selected.final_score, 6),
        },
        "used_fallback": fallback,
    }


def _serialize_output(output: Any) -> str:
    return json.dumps(output, ensure_ascii=False, sort_keys=True)
