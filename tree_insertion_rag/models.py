from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


@dataclass(slots=True)
class ParsedNode:
    node_id: str
    node_name: str
    node_type: str
    annotation: str
    jsonpath: str
    depth: int
    ancestor_path_names: list[str] = field(default_factory=list)
    ancestor_path_ids: list[str] = field(default_factory=list)
    children_summary: list[str] = field(default_factory=list)
    children_texts: list[str] = field(default_factory=list)
    raw_node: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateNode:
    node_id: str
    node_name: str
    node_type: str
    annotation: str
    jsonpath: str
    depth: int
    ancestor_path_names: list[str] = field(default_factory=list)
    children_summary: list[str] = field(default_factory=list)
    children_texts: list[str] = field(default_factory=list)
    retrieval_text: str = ""
    raw_node: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RankedCandidate:
    candidate: CandidateNode
    semantic_score: float
    sibling_score: float
    path_prior_score: float
    final_score: float
    reason: str


@dataclass(slots=True)
class PathPriorRule:
    name: str
    trigger_terms: list[str]
    boost_terms: list[str]


@dataclass(slots=True)
class RetrievalHit:
    candidate: CandidateNode
    sparse_score: float
    dense_score: float
    hybrid_score: float


class EmbeddingProvider(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class TreeInsertionError(ValueError):
    """Raised when the tree structure or input payload is invalid."""


class SearchAction(str, Enum):
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
