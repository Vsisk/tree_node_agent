from __future__ import annotations

import json
import os
from pathlib import Path

from tree_insertion_rag.ranker import Ranker
from tree_insertion_rag.selector import LLMCandidateSelector, TreeInsertionSelector


def build_demo_tree() -> dict:
    return {
        "mapping_content": {
            "node_name": "mapping_content",
            "node_id": "root",
            "node_type": "parent",
            "annotation": "invoice mapping root",
            "children": [
                {
                    "node_name": "basic_info",
                    "node_id": "p_basic",
                    "node_type": "parent",
                    "annotation": "invoice header information",
                    "children": [
                        {
                            "node_name": "invoice_no",
                            "node_id": "l_invoice_no",
                            "node_type": "leaf",
                            "annotation": "invoice number",
                        },
                        {
                            "node_name": "invoice_date",
                            "node_id": "l_invoice_date",
                            "node_type": "leaf",
                            "annotation": "invoice issue date",
                        },
                    ],
                },
                {
                    "node_name": "fee_detail",
                    "node_id": "p_fee",
                    "node_type": "parent",
                    "annotation": "amount and tax fields",
                    "children": [
                        {
                            "node_name": "amount",
                            "node_id": "l_amount",
                            "node_type": "leaf",
                            "annotation": "total amount",
                        },
                        {
                            "node_name": "tax",
                            "node_id": "l_tax",
                            "node_type": "leaf",
                            "annotation": "tax amount",
                        },
                        {
                            "node_name": "currency",
                            "node_id": "l_currency",
                            "node_type": "leaf",
                            "annotation": "currency code",
                        },
                    ],
                },
                {
                    "node_name": "appendix",
                    "node_id": "p_appendix",
                    "node_type": "parent",
                    "annotation": "remarks and attachments",
                    "children": [
                        {
                            "node_name": "remark",
                            "node_id": "l_remark",
                            "node_type": "leaf",
                            "annotation": "business remark",
                        }
                    ],
                },
            ],
        }
    }


def build_demo_target() -> dict:
    return {
        "node_name": "service_fee",
        "node_id": "n_service_fee",
        "node_type": "leaf",
        "annotation": "service fee amount",
    }


def build_demo_query() -> str:
    return "service fee belongs with amount and tax in the fee detail section"


def build_selector() -> TreeInsertionSelector:
    load_project_env()
    llm_mode = os.getenv("TREE_INSERTION_DEMO_LLM", "").strip().lower()
    ranker = Ranker()

    if llm_mode != "openai":
        return TreeInsertionSelector(ranker=ranker)

    candidate_selector = LLMCandidateSelector(build_openai_callable())
    return TreeInsertionSelector(ranker=ranker, candidate_selector=candidate_selector)


def build_openai_callable():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai package is required when TREE_INSERTION_DEMO_LLM=openai") from exc

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when TREE_INSERTION_DEMO_LLM=openai")

    model = (
        os.getenv("TREE_INSERTION_DEMO_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or "gpt-4.1-mini"
    )
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    timeout = _parse_timeout(os.getenv("OPENAI_TIMEOUT", "").strip())

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    if timeout is not None:
        client_kwargs["timeout"] = timeout

    client = OpenAI(**client_kwargs)

    def _call(prompt: str, ranked_candidates):
        del ranked_candidates
        response = client.responses.create(
            model=model,
            input=prompt,
        )
        return response.output_text.strip()

    return _call


def load_project_env() -> None:
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        _load_env_without_dependency(env_path)
        return

    load_dotenv(env_path, override=False)


def _load_env_without_dependency(env_path: Path) -> None:
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_timeout(timeout_value: str) -> float | None:
    if not timeout_value:
        return None
    try:
        return float(timeout_value)
    except ValueError as exc:
        raise RuntimeError(f"OPENAI_TIMEOUT must be numeric, got: {timeout_value}") from exc


def main() -> None:
    try:
        selector = build_selector()
        jsonpath = selector.find_best_node(
            tree=build_demo_tree(),
            query=build_demo_query(),
            action="add",
            node=build_demo_target(),
            topk=5,
        )
    except RuntimeError as exc:
        raise SystemExit(
            "Demo failed to start. Install `sentence_transformers` for BGE-M3, "
            "and install `openai` if you set TREE_INSERTION_DEMO_LLM=openai. "
            f"Original error: {exc}"
        ) from exc

    print(json.dumps({"jsonpath": jsonpath}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
