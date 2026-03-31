from __future__ import annotations

from tree_insertion_rag.models import CandidateNode, ParsedNode, SearchAction
from tree_insertion_rag.utils import summarize_text


class CandidateBuilder:
    """Build parent-node candidates and their retrieval text."""

    def build(
        self,
        parsed_nodes: list[ParsedNode],
        action: SearchAction = SearchAction.ADD,
    ) -> list[CandidateNode]:
        candidates: list[CandidateNode] = []
        for node in parsed_nodes:
            if not self._is_candidate_for_action(node, action):
                continue
            retrieval_text = self._build_retrieval_text(node, action)
            candidates.append(
                CandidateNode(
                    node_id=node.node_id,
                    node_name=node.node_name,
                    node_type=node.node_type,
                    jsonpath=node.jsonpath,
                    annotation=node.annotation,
                    depth=node.depth,
                    ancestor_path_names=list(node.ancestor_path_names),
                    children_summary=list(node.children_summary),
                    children_texts=list(node.children_texts),
                    retrieval_text=retrieval_text,
                    raw_node=node.raw_node,
                )
            )
        return candidates

    @staticmethod
    def _is_insertable_parent(node: ParsedNode) -> bool:
        return node.node_type == "parent" and isinstance(node.raw_node.get("children"), list)

    def _is_candidate_for_action(self, node: ParsedNode, action: SearchAction) -> bool:
        if action == SearchAction.ADD:
            return self._is_insertable_parent(node)
        return True

    def _build_retrieval_text(self, node: ParsedNode, action: SearchAction) -> str:
        ancestors = " > ".join(node.ancestor_path_names) if node.ancestor_path_names else "<ROOT>"
        children = ", ".join(node.children_summary[:8]) if node.children_summary else "<EMPTY>"
        responsibility = self._build_responsibility_text(node, action)
        return "\n".join(
            [
                f"当前节点名称: {node.node_name}",
                f"当前节点说明: {node.annotation}",
                f"当前节点类型: {node.node_type}",
                f"祖先路径: {ancestors}",
                f"直接子节点摘要: {children}",
                f"容器职责说明: {responsibility}",
            ]
        )

    def _build_responsibility_text(self, node: ParsedNode, action: SearchAction) -> str:
        if action != SearchAction.ADD and node.node_type != "parent":
            if node.ancestor_path_names:
                trail = " > ".join(node.ancestor_path_names[-2:])
                return f"该节点是 {trail} 路径下的具体业务字段，可用于 modify 或 delete 精确定位。"
            return "该节点是具体业务字段，可用于 modify 或 delete 精确定位。"
        if node.children_summary:
            preview = ", ".join(node.children_summary[:3])
            return f"该父节点负责容纳与 {preview} 同域或同类的子节点。"
        if node.annotation:
            return f"该节点负责承接与 {summarize_text(node.annotation, 32)} 相关的内容。"
        if node.ancestor_path_names:
            trail = " > ".join(node.ancestor_path_names[-2:])
            return f"该节点位于 {trail} 路径下，适合作为局部命中点。"
        return "该节点是树中的可检索结构单元。"
