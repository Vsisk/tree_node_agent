from __future__ import annotations

from typing import Any

from tree_insertion_rag.models import SearchAction, TreeInsertionError
from tree_insertion_rag.utils import ensure_str


class QueryBuilder:
    """Combine the target node and the external query into a retrieval query."""

    REQUIRED_TARGET_FIELDS = {"node_name", "node_id", "node_type", "annotation"}

    def build(
        self,
        query: str,
        action: SearchAction,
        target_node: dict[str, Any] | None = None,
    ) -> str:
        if target_node is not None:
            self._validate_target_node(target_node)

        lines = [
            f"动作类型: {action.value}",
            f"补充意图: {ensure_str(query)}",
        ]

        if target_node is not None:
            lines.extend(
                [
                    f"目标节点名称: {ensure_str(target_node.get('node_name'))}",
                    f"目标节点说明: {ensure_str(target_node.get('annotation'))}",
                    f"目标节点类型: {ensure_str(target_node.get('node_type'))}",
                ]
            )
        else:
            lines.append("目标节点名称: <UNKNOWN>")

        if action == SearchAction.ADD:
            lines.append("检索目标: 寻找最适合作为父节点的已有 parent 节点，并确保该位置可继续插入 children。")
        elif action == SearchAction.MODIFY:
            lines.append("检索目标: 在整棵树中定位最适合被修改的具体节点，候选范围包括 parent 和 leaf。")
        else:
            lines.append("检索目标: 在整棵树中定位最适合被删除的具体节点，候选范围包括 parent 和 leaf。")

        return "\n".join(lines)

    def _validate_target_node(self, target_node: dict[str, Any]) -> None:
        if not isinstance(target_node, dict):
            raise TreeInsertionError("target_node must be a dictionary")
        missing = [field for field in self.REQUIRED_TARGET_FIELDS if field not in target_node]
        if missing:
            raise TreeInsertionError(
                f"target_node is missing required fields: {', '.join(sorted(missing))}"
            )
