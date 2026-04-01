from __future__ import annotations

from typing import Any, Protocol

from tree_insertion_rag.parser import TreeInsertionError, TreeParser
from tree_insertion_rag.ranker import Ranker, RankedCandidate


class CandidateSelector(Protocol):
    def select(
        self,
        query: str,
        action: str,
        target_node: dict[str, Any] | None,
        ranked_candidates: list[RankedCandidate],
    ) -> RankedCandidate | None:
        ...


class LLMCandidateSelector:
    """Use an LLM callable to choose the final candidate from the ranked shortlist."""

    def __init__(self, llm_callable: Any) -> None:
        self.llm_callable = llm_callable

    def select(
        self,
        query: str,
        action: str,
        target_node: dict[str, Any] | None,
        ranked_candidates: list[RankedCandidate],
    ) -> RankedCandidate | None:
        if not ranked_candidates:
            return None
        prompt = self._build_prompt(query, action, target_node, ranked_candidates)
        result = self.llm_callable(prompt, ranked_candidates)
        if isinstance(result, RankedCandidate):
            return result
        selected_id = str(result).strip() if result is not None else ""
        if not selected_id:
            return ranked_candidates[0]
        for candidate in ranked_candidates:
            if candidate.candidate.node_id == selected_id or candidate.candidate.jsonpath == selected_id:
                return candidate
        return ranked_candidates[0]

    @staticmethod
    def _build_prompt(
        query: str,
        action: str,
        target_node: dict[str, Any] | None,
        ranked_candidates: list[RankedCandidate],
    ) -> str:
        lines = [
            "Choose the single best candidate.",
            f"action: {action}",
            f"query: {query}",
        ]
        if target_node is not None:
            lines.extend(
                [
                    f"target_name: {target_node.get('node_name', '')}",
                    f"target_type: {target_node.get('node_type', '')}",
                    f"target_annotation: {target_node.get('annotation', '')}",
                ]
            )
        lines.append("candidates:")
        for index, ranked in enumerate(ranked_candidates, start=1):
            candidate = ranked.candidate
            lines.append(
                f"{index}. node_id={candidate.node_id}; jsonpath={candidate.jsonpath}; "
                f"name={candidate.node_name}; type={candidate.node_type}; score={ranked.final_score:.4f}"
            )
        lines.append("Return only node_id or jsonpath.")
        return "\n".join(lines)


class TreeInsertionSelector:
    """Facade with unchanged inputs and a single jsonpath output."""

    def __init__(
        self,
        ranker: Ranker | None = None,
        candidate_selector: CandidateSelector | None = None,
    ) -> None:
        self.parser = TreeParser()
        self.ranker = ranker or Ranker()
        self.candidate_selector = candidate_selector

    def find_best_node(
        self,
        tree: dict[str, Any],
        query: str,
        action: str,
        node: dict[str, Any] | None = None,
        topk: int = 10,
        verbose: bool | None = None,
    ) -> str | None:
        del verbose
        normalized_action = self._normalize_action(action)
        if normalized_action == "add" and node is None:
            raise TreeInsertionError("node is required when action='add'")

        parsed_nodes = self.parser.parse(tree)
        ranked_candidates = self.ranker.rank(
            parsed_nodes=parsed_nodes,
            query=query,
            action=normalized_action,
            target_node=node,
            topk=topk,
        )
        if not ranked_candidates:
            return None

        if self.candidate_selector is None:
            return ranked_candidates[0].candidate.jsonpath

        selected = self.candidate_selector.select(
            query=query,
            action=normalized_action,
            target_node=node,
            ranked_candidates=ranked_candidates,
        )
        if selected is None:
            return ranked_candidates[0].candidate.jsonpath
        return selected.candidate.jsonpath

    def find_best_parent(
        self,
        tree: dict[str, Any],
        target_node: dict[str, Any],
        query: str,
        topk: int = 10,
    ) -> str | None:
        return self.find_best_node(
            tree=tree,
            query=query,
            action="add",
            node=target_node,
            topk=topk,
        )

    @staticmethod
    def _normalize_action(action: str) -> str:
        normalized = action.strip().lower()
        if normalized not in {"add", "modify", "delete"}:
            raise TreeInsertionError("action must be one of: add, modify, delete")
        return normalized
