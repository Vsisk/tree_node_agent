from __future__ import annotations

import logging
from typing import Any

from tree_insertion_rag.candidate_builder import CandidateBuilder
from tree_insertion_rag.index_manager import EmbeddingProvider, IndexManager
from tree_insertion_rag.models import SearchAction, TreeInsertionError
from tree_insertion_rag.query_builder import QueryBuilder
from tree_insertion_rag.ranker import Ranker
from tree_insertion_rag.tracing import PipelineTracer
from tree_insertion_rag.tree_parser import TreeParser


class TreeInsertionRetriever:
    """Facade for parsing, indexing, retrieval and ranking."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        verbose: bool = False,
        logger: logging.Logger | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.verbose = verbose
        self.logger = logger or logging.getLogger("tree_insertion_rag.retriever")
        self.log_level = log_level
        self.tree_parser = TreeParser()
        self.candidate_builder = CandidateBuilder()
        self.query_builder = QueryBuilder()
        self.index_manager = IndexManager(embedding_provider=embedding_provider)
        self.ranker = Ranker(embedding_provider=self.index_manager.embedding_provider)
        self.last_debug: dict[str, Any] = {}

    def build_index(
        self,
        tree: dict[str, Any],
        action: SearchAction = SearchAction.ADD,
        tracer: PipelineTracer | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        parsed_nodes = self.tree_parser.parse(tree)
        candidates = self.candidate_builder.build(parsed_nodes, action=action)
        self.index_manager.build(candidates, tracer=tracer)
        self.ranker.embedding_provider = self.index_manager.embedding_provider
        parsed_debug = [
            {
                "node_id": node.node_id,
                "node_name": node.node_name,
                "node_type": node.node_type,
                "jsonpath": node.jsonpath,
                "depth": node.depth,
            }
            for node in parsed_nodes
        ]
        candidate_debug = [
            {
                "node_id": candidate.node_id,
                "node_name": candidate.node_name,
                "node_type": candidate.node_type,
                "jsonpath": candidate.jsonpath,
                "depth": candidate.depth,
            }
            for candidate in candidates
        ]
        if tracer is not None:
            tracer.log(
                "tree.parse",
                "Parsed tree nodes",
                {
                    "parsed_count": len(parsed_debug),
                    "sample_nodes": parsed_debug[:8],
                },
            )
            tracer.log(
                "candidate.build",
                "Built retrieval candidates",
                {
                    "action": action.value,
                    "candidate_count": len(candidate_debug),
                    "sample_candidates": candidate_debug[:8],
                },
            )
        return parsed_debug, candidate_debug

    def find_best_node(
        self,
        tree: dict[str, Any],
        query: str,
        action: str | SearchAction,
        node: dict[str, Any] | None = None,
        topk: int = 10,
        verbose: bool | None = None,
    ) -> dict[str, Any]:
        resolved_action = self._normalize_action(action)
        self._validate_action_payload(resolved_action, node)
        tracer = PipelineTracer(
            enabled=self.verbose if verbose is None else verbose,
            logger=self.logger,
            level=self.log_level,
        )
        tracer.log(
            "request.start",
            "Starting retrieval request",
            {
                "action": resolved_action.value,
                "topk": topk,
                "input_node_id": node.get("node_id") if node else None,
                "query": query,
            },
        )

        parsed_debug, candidate_debug = self.build_index(tree, action=resolved_action, tracer=tracer)
        query_text = self.query_builder.build(query=query, action=resolved_action, target_node=node)
        tracer.log(
            "query.build",
            "Built retrieval query",
            {
                "query_text": query_text,
            },
        )
        recall_topk = max(topk * 3, topk)
        retrieved_candidates = self.index_manager.hybrid_retrieve(query_text, recall_topk, tracer=tracer)
        ranked_candidates = self.ranker.rank(
            query_text=query_text,
            target_node=node,
            candidates=retrieved_candidates,
            topk=topk,
            action=resolved_action,
            tracer=tracer,
        )
        confidence, confidence_reason = self.ranker.classify_confidence(
            ranked_candidates=ranked_candidates,
            action=resolved_action,
            target_node=node,
        )
        tracer.log(
            "confidence.final",
            "Classified final confidence",
            {
                "confidence": confidence,
                "reason": confidence_reason,
                "best_score": round(ranked_candidates[0].final_score, 4) if ranked_candidates else 0.0,
            },
        )

        if not ranked_candidates:
            self.last_debug = {
                "query_text": query_text,
                "parsed_nodes": parsed_debug,
                "candidates": candidate_debug,
                "index": self.index_manager.last_build_debug,
                "retrieval": self.index_manager.last_retrieval_debug,
                "ranking": self.ranker.last_rank_debug,
                "trace": tracer.export(),
            }
            return {
                "action": resolved_action.value,
                "jsonpath": None,
                "matched_node_id": None,
                "input_node_id": node.get("node_id") if node else None,
                "score": 0.0,
                "reason": confidence_reason,
                "confidence": "low",
                "top_candidates": [],
                "debug": self.last_debug,
            }

        best = ranked_candidates[0]
        should_return_path = confidence != "low" or resolved_action == SearchAction.ADD
        if confidence == "low" and resolved_action == SearchAction.ADD:
            reason = f"{confidence_reason}; fallback to top candidate for add"
        else:
            reason = best.reason if should_return_path else confidence_reason
        self.last_debug = {
            "query_text": query_text,
            "parsed_nodes": parsed_debug,
            "candidates": candidate_debug,
            "index": self.index_manager.last_build_debug,
            "retrieval": self.index_manager.last_retrieval_debug,
            "ranking": self.ranker.last_rank_debug,
            "trace": tracer.export(),
        }
        return {
            "action": resolved_action.value,
            "jsonpath": best.candidate.jsonpath if should_return_path else None,
            "matched_node_id": best.candidate.node_id if should_return_path else None,
            "input_node_id": node.get("node_id") if node else None,
            "score": round(best.final_score, 4),
            "reason": reason,
            "confidence": confidence,
            "top_candidates": [
                {
                    "jsonpath": ranked.candidate.jsonpath,
                    "node_id": ranked.candidate.node_id,
                    "score": round(ranked.final_score, 4),
                    "reason": ranked.reason,
                }
                for ranked in ranked_candidates
            ],
            "debug": self.last_debug,
        }

    def find_best_parent(
        self,
        tree: dict[str, Any],
        target_node: dict[str, Any],
        query: str,
        topk: int = 10,
    ) -> dict[str, Any]:
        return self.find_best_node(
            tree=tree,
            query=query,
            action=SearchAction.ADD,
            node=target_node,
            topk=topk,
        )

    @staticmethod
    def _normalize_action(action: str | SearchAction) -> SearchAction:
        if isinstance(action, SearchAction):
            return action
        try:
            return SearchAction(action.lower())
        except Exception as exc:  # pragma: no cover - defensive branch
            raise TreeInsertionError("action must be one of: add, modify, delete") from exc

    @staticmethod
    def _validate_action_payload(action: SearchAction, node: dict[str, Any] | None) -> None:
        if action == SearchAction.ADD and node is None:
            raise TreeInsertionError("node is required when action='add'")
