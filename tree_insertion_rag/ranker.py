from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import re
from typing import Any, Protocol

from tree_insertion_rag.parser import ParsedNode, build_node_text

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None


class EmbeddingModel(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(slots=True)
class CandidateNode:
    node_id: str
    node_name: str
    node_type: str
    annotation: str
    jsonpath: str
    ancestor_names: list[str] = field(default_factory=list)
    children_texts: list[str] = field(default_factory=list)
    ranking_text: str = ""


@dataclass(slots=True)
class RankedCandidate:
    candidate: CandidateNode
    semantic_score: float
    sibling_score: float
    path_prior_score: float
    final_score: float


class BgeM3EmbeddingModel:
    """BGE-M3 embedding wrapper without fallback providers."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or os.getenv("BGE_M3_DEVICE")
        self._model: Any = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if SentenceTransformer is None:
            raise RuntimeError("sentence_transformers is required to use BAAI/bge-m3")
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


class Ranker:
    """Build candidates and rank them with BGE-M3 embeddings plus light structural priors."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        alpha: float = 0.75,
        beta: float = 0.15,
        gamma: float = 0.10,
    ) -> None:
        self.embedding_model = embedding_model or BgeM3EmbeddingModel()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def rank(
        self,
        parsed_nodes: list[ParsedNode],
        query: str,
        action: str,
        target_node: dict[str, Any] | None = None,
        topk: int = 10,
    ) -> list[RankedCandidate]:
        candidates = self._build_candidates(parsed_nodes, action)
        if not candidates:
            return []

        query_text = self._build_query_text(query=query, action=action, target_node=target_node)
        query_vector = self.embedding_model.encode([query_text])[0]
        candidate_vectors = self.embedding_model.encode([candidate.ranking_text for candidate in candidates])
        target_text = build_node_text(
            target_node.get("node_name", "") if target_node else "",
            target_node.get("annotation", "") if target_node else "",
        )

        ranked: list[RankedCandidate] = []
        for candidate, candidate_vector in zip(candidates, candidate_vectors):
            semantic_score = cosine_similarity(query_vector, candidate_vector)
            sibling_score = self._compute_sibling_score(candidate, target_text, action)
            path_prior_score = self._compute_path_prior_score(candidate, query)
            final_score = self.alpha * semantic_score + self.beta * sibling_score + self.gamma * path_prior_score
            ranked.append(
                RankedCandidate(
                    candidate=candidate,
                    semantic_score=semantic_score,
                    sibling_score=sibling_score,
                    path_prior_score=path_prior_score,
                    final_score=final_score,
                )
            )

        ranked.sort(key=lambda item: item.final_score, reverse=True)
        return ranked[:topk]

    def _build_candidates(self, parsed_nodes: list[ParsedNode], action: str) -> list[CandidateNode]:
        include_only_parents = action == "add"
        candidates: list[CandidateNode] = []
        for node in parsed_nodes:
            if include_only_parents and not self._is_insertable_parent(node):
                continue
            candidates.append(
                CandidateNode(
                    node_id=node.node_id,
                    node_name=node.node_name,
                    node_type=node.node_type,
                    annotation=node.annotation,
                    jsonpath=node.jsonpath,
                    ancestor_names=list(node.ancestor_names),
                    children_texts=list(node.children_texts),
                    ranking_text=self._build_candidate_text(node),
                )
            )
        return candidates

    @staticmethod
    def _is_insertable_parent(node: ParsedNode) -> bool:
        return node.node_type == "parent" and isinstance(node.raw_node.get("children"), list)

    @staticmethod
    def _build_query_text(query: str, action: str, target_node: dict[str, Any] | None) -> str:
        lines = [f"action: {action}", f"query: {query.strip()}"]
        if target_node is not None:
            lines.extend(
                [
                    f"target_name: {target_node.get('node_name', '')}",
                    f"target_type: {target_node.get('node_type', '')}",
                    f"target_annotation: {target_node.get('annotation', '')}",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _build_candidate_text(node: ParsedNode) -> str:
        ancestors = " > ".join(node.ancestor_names)
        children = " | ".join(text for text in node.children_texts if text)
        return "\n".join(
            [
                f"name: {node.node_name}",
                f"type: {node.node_type}",
                f"annotation: {node.annotation}",
                f"ancestors: {ancestors}",
                f"children: {children}",
            ]
        )

    @staticmethod
    def _compute_sibling_score(candidate: CandidateNode, target_text: str, action: str) -> float:
        if not target_text:
            return 0.0
        if action == "add":
            if not candidate.children_texts:
                return 0.0
            return max(cosine_similarity_from_text(target_text, child_text) for child_text in candidate.children_texts)
        return cosine_similarity_from_text(target_text, build_node_text(candidate.node_name, candidate.annotation))

    @staticmethod
    def _compute_path_prior_score(candidate: CandidateNode, query: str) -> float:
        normalized_query = normalize_text(query)
        if not normalized_query:
            return 0.0

        score = 0.0
        name = normalize_text(candidate.node_name)
        annotation = normalize_text(candidate.annotation)
        if name and name in normalized_query:
            score += 0.6
        if annotation and annotation in normalized_query:
            score += 0.2
        ancestor_hits = sum(1 for ancestor in candidate.ancestor_names if normalize_text(ancestor) in normalized_query)
        score += min(ancestor_hits * 0.1, 0.2)
        return min(score, 1.0)


def normalize_text(text: str) -> str:
    normalized = text.lower().replace("_", " ")
    normalized = re.sub(r"[\r\n\t]+", " ", normalized)
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    ascii_tokens = re.findall(r"[a-z0-9]+", normalized)
    cjk_segments = re.findall(r"[\u4e00-\u9fff]+", normalized)
    cjk_tokens: list[str] = []
    for segment in cjk_segments:
        cjk_tokens.extend(list(segment))
        cjk_tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return ascii_tokens + cjk_tokens


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def cosine_similarity_from_text(left: str, right: str) -> float:
    left_counts = token_counts(left)
    right_counts = token_counts(right)
    if not left_counts or not right_counts:
        return 0.0
    numerator = sum(left_counts[token] * right_counts.get(token, 0.0) for token in left_counts)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def token_counts(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in tokenize(text):
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts
