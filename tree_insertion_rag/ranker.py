from __future__ import annotations

from typing import Any

from tree_insertion_rag.models import CandidateNode, EmbeddingProvider, PathPriorRule, RankedCandidate, SearchAction
from tree_insertion_rag.tracing import PipelineTracer
from tree_insertion_rag.utils import (
    average_top_scores,
    build_node_text,
    cosine_similarity,
    lexical_similarity,
    normalize_text,
)


DEFAULT_PATH_PRIOR_RULES = [
    PathPriorRule(
        name="basic_info",
        trigger_terms=["基础信息", "基本信息", "概况", "overview", "basic information"],
        boost_terms=["基础信息", "基本信息", "概况", "overview", "header", "general info"],
    ),
    PathPriorRule(
        name="fees",
        trigger_terms=["费用", "税费", "金额", "billing", "charge", "tax", "amount", "payment"],
        boost_terms=["费用", "税费", "金额", "billing", "charge", "tax", "payment", "price", "fee"],
    ),
    PathPriorRule(
        name="summary",
        trigger_terms=["摘要", "汇总", "summary", "total"],
        boost_terms=["摘要", "汇总", "summary", "total", "overview"],
    ),
    PathPriorRule(
        name="appendix",
        trigger_terms=["附录", "附件", "appendix", "attachment"],
        boost_terms=["附录", "附件", "appendix", "attachment"],
    ),
]


class Ranker:
    """Re-rank retrieved candidates with structural heuristics."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        alpha: float = 0.60,
        beta: float = 0.25,
        gamma: float = 0.15,
        min_confidence_score: float = 0.38,
        min_margin: float = 0.08,
        path_prior_rules: list[PathPriorRule] | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_confidence_score = min_confidence_score
        self.min_margin = min_margin
        self.path_prior_rules = path_prior_rules or list(DEFAULT_PATH_PRIOR_RULES)
        self.last_rank_debug: dict[str, Any] = {}

    def rank(
        self,
        query_text: str,
        target_node: dict[str, Any] | None,
        candidates: list[CandidateNode],
        topk: int,
        action: SearchAction = SearchAction.ADD,
        tracer: PipelineTracer | None = None,
    ) -> list[RankedCandidate]:
        if not candidates:
            self.last_rank_debug = {"action": action.value, "candidate_count": 0, "ranked_candidates": []}
            return []

        query_vector = self.embedding_provider.encode([query_text])[0]
        candidate_vectors = self.embedding_provider.encode([candidate.retrieval_text for candidate in candidates])
        target_text = (
            build_node_text(target_node.get("node_name", ""), target_node.get("annotation", ""))
            if target_node is not None
            else ""
        )

        ranked: list[RankedCandidate] = []
        for candidate, candidate_vector in zip(candidates, candidate_vectors):
            if not self._is_candidate_allowed(candidate, action):
                continue
            semantic_score = self._compute_semantic_score(query_text, candidate, query_vector, candidate_vector)
            sibling_score = self._compute_sibling_score(target_text, candidate, action)
            path_prior_score, matched_rule_names = self._compute_path_prior_score(query_text, candidate)
            final_score = (
                self.alpha * semantic_score
                + self.beta * sibling_score
                + self.gamma * path_prior_score
            )
            reason = self._build_reason(
                candidate=candidate,
                action=action,
                semantic_score=semantic_score,
                sibling_score=sibling_score,
                path_prior_score=path_prior_score,
                matched_rule_names=matched_rule_names,
            )
            ranked.append(
                RankedCandidate(
                    candidate=candidate,
                    semantic_score=semantic_score,
                    sibling_score=sibling_score,
                    path_prior_score=path_prior_score,
                    final_score=final_score,
                    reason=reason,
                )
            )

        ranked.sort(key=lambda item: item.final_score, reverse=True)
        final_ranked = ranked[:topk]
        self.last_rank_debug = {
            "action": action.value,
            "candidate_count": len(candidates),
            "ranked_candidates": [
                {
                    "node_id": item.candidate.node_id,
                    "node_name": item.candidate.node_name,
                    "jsonpath": item.candidate.jsonpath,
                    "semantic_score": round(item.semantic_score, 4),
                    "sibling_score": round(item.sibling_score, 4),
                    "path_prior_score": round(item.path_prior_score, 4),
                    "final_score": round(item.final_score, 4),
                    "reason": item.reason,
                }
                for item in final_ranked
            ],
        }
        if tracer is not None:
            tracer.log(
                "rank.final",
                "Ranking completed",
                {
                    "action": action.value,
                    "top_ranked": self.last_rank_debug["ranked_candidates"][:5],
                },
            )
        return final_ranked

    def classify_confidence(
        self,
        ranked_candidates: list[RankedCandidate],
        action: SearchAction = SearchAction.ADD,
        target_node: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        if not ranked_candidates:
            if action == SearchAction.ADD:
                return "low", "当前树中没有可作为插入位置的 parent 候选节点。"
            return "low", "当前树中没有足够可靠的候选节点。"

        best = ranked_candidates[0]
        second_score = ranked_candidates[1].final_score if len(ranked_candidates) > 1 else 0.0
        margin = best.final_score - second_score
        min_score, min_margin = self._resolve_confidence_thresholds(action, target_node)

        if best.final_score < min_score:
            if action == SearchAction.ADD:
                return "low", "当前树中没有足够可靠的候选父节点。"
            return "low", "当前树中没有足够可靠的候选节点。"
        if margin < min_margin:
            if best.final_score >= min_score + 0.12:
                return "medium", "第一名候选有效，但与第二名差距较小。"
            if action == SearchAction.ADD:
                return "low", "候选之间分差过小，无法可靠判断唯一父节点。"
            return "low", "候选之间分差过小，无法可靠判断唯一目标节点。"
        if best.final_score >= 0.72:
            return "high", "该位置语义、兄弟模式和路径先验都较为一致。"
        return "medium", "该位置可行，但仍建议结合业务做一次人工确认。"

    def _resolve_confidence_thresholds(
        self,
        action: SearchAction,
        target_node: dict[str, Any] | None,
    ) -> tuple[float, float]:
        if action == SearchAction.ADD:
            return self.min_confidence_score, self.min_margin
        if target_node is None:
            return 0.18, 0.01
        return 0.22, 0.02

    @staticmethod
    def _is_insertable_parent(candidate: CandidateNode) -> bool:
        return candidate.node_type == "parent" and isinstance(candidate.raw_node.get("children"), list)

    def _is_candidate_allowed(self, candidate: CandidateNode, action: SearchAction) -> bool:
        if action == SearchAction.ADD:
            return self._is_insertable_parent(candidate)
        return True

    def _compute_semantic_score(
        self,
        query_text: str,
        candidate: CandidateNode,
        query_vector: list[float],
        candidate_vector: list[float],
    ) -> float:
        dense_score = cosine_similarity(query_vector, candidate_vector)
        lexical_score = lexical_similarity(query_text, candidate.retrieval_text)
        return 0.7 * dense_score + 0.3 * lexical_score

    def _compute_sibling_score(
        self,
        target_text: str,
        candidate: CandidateNode,
        action: SearchAction,
    ) -> float:
        if not target_text:
            return 0.0
        if action == SearchAction.ADD:
            if not candidate.children_texts:
                return 0.0
            child_scores = [lexical_similarity(target_text, child_text) for child_text in candidate.children_texts]
            return average_top_scores(child_scores, topn=2)

        self_score = lexical_similarity(target_text, build_node_text(candidate.node_name, candidate.annotation))
        sibling_scores = [lexical_similarity(target_text, child_text) for child_text in candidate.children_texts]
        sibling_hint = max(sibling_scores) if sibling_scores else 0.0
        return 0.8 * self_score + 0.2 * sibling_hint

    def _compute_path_prior_score(
        self,
        query_text: str,
        candidate: CandidateNode,
    ) -> tuple[float, list[str]]:
        query_normalized = normalize_text(query_text)
        candidate_context = normalize_text(
            " ".join(candidate.ancestor_path_names + [candidate.node_name, candidate.annotation])
        )
        active_rules = [
            rule
            for rule in self.path_prior_rules
            if any(normalize_text(term) and normalize_text(term) in query_normalized for term in rule.trigger_terms)
        ]

        mention_score = self._compute_query_mention_score(query_normalized, candidate)
        if not active_rules:
            return mention_score, []

        matched_rule_names: list[str] = []
        scores: list[float] = []
        for rule in active_rules:
            substring_match = any(
                normalize_text(term) and normalize_text(term) in candidate_context for term in rule.boost_terms
            )
            if substring_match:
                matched_rule_names.append(rule.name)
                scores.append(1.0)
                continue
            rule_hint_text = " ".join(rule.boost_terms)
            approx_score = lexical_similarity(rule_hint_text, candidate_context)
            if approx_score >= 0.22:
                matched_rule_names.append(rule.name)
            scores.append(min(approx_score, 1.0))

        rule_score = sum(scores) / len(scores)
        return min(1.0, 0.6 * rule_score + 0.4 * mention_score), matched_rule_names

    def _compute_query_mention_score(self, query_normalized: str, candidate: CandidateNode) -> float:
        if not query_normalized:
            return 0.0
        score = 0.0
        candidate_name = normalize_text(candidate.node_name)
        candidate_annotation = normalize_text(candidate.annotation)
        if candidate_name and candidate_name in query_normalized:
            score += 0.6
        if candidate_annotation and candidate_annotation in query_normalized:
            score += 0.2
        ancestor_hits = 0
        for ancestor_name in candidate.ancestor_path_names:
            normalized_ancestor = normalize_text(ancestor_name)
            if normalized_ancestor and normalized_ancestor in query_normalized:
                ancestor_hits += 1
        if ancestor_hits:
            score += min(0.2 * ancestor_hits, 0.4)
        return min(score, 1.0)

    def _build_reason(
        self,
        candidate: CandidateNode,
        action: SearchAction,
        semantic_score: float,
        sibling_score: float,
        path_prior_score: float,
        matched_rule_names: list[str],
    ) -> str:
        fragments = [
            (
                f"候选节点 {candidate.node_name} 是可插入 parent"
                if action == SearchAction.ADD
                else f"候选节点 {candidate.node_name} 是可操作节点"
            ),
            f"语义匹配分 {semantic_score:.2f}",
            f"兄弟相似分 {sibling_score:.2f}",
            f"路径先验分 {path_prior_score:.2f}",
        ]
        if candidate.children_summary:
            preview = ", ".join(candidate.children_summary[:3])
            fragments.append(f"children 中已有同域节点：{preview}")
        if matched_rule_names:
            fragments.append(f"命中路径先验规则：{', '.join(matched_rule_names)}")
        return "；".join(fragments) + "。"
