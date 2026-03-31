from __future__ import annotations

from typing import Any

from tree_insertion_rag.models import ParsedNode, TreeInsertionError
from tree_insertion_rag.utils import build_node_text, ensure_str, has_required_node_fields, validate_node_payload


class TreeParser:
    """Parse a tree JSON into flattened nodes with stable jsonpath metadata."""

    def parse(self, tree: dict[str, Any]) -> list[ParsedNode]:
        root_node, root_path = self._resolve_root(tree)
        parsed_nodes: list[ParsedNode] = []
        self._walk(
            node=root_node,
            jsonpath=root_path,
            depth=0,
            ancestor_names=[],
            ancestor_ids=[],
            output=parsed_nodes,
        )
        return parsed_nodes

    def _resolve_root(self, tree: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if has_required_node_fields(tree):
            validate_node_payload(tree, allow_children_missing=False)
            return tree, "$"

        if not isinstance(tree, dict) or not tree:
            raise TreeInsertionError("tree must be a non-empty dictionary")

        if len(tree) != 1:
            raise TreeInsertionError(
                "wrapped tree input must contain exactly one root field, for example {'mapping_content': {...}}"
            )

        root_key, root_value = next(iter(tree.items()))
        if not has_required_node_fields(root_value):
            raise TreeInsertionError("wrapped tree root must be a valid node object")

        validate_node_payload(root_value, allow_children_missing=False)
        return root_value, f"$.{root_key}"

    def _walk(
        self,
        node: dict[str, Any],
        jsonpath: str,
        depth: int,
        ancestor_names: list[str],
        ancestor_ids: list[str],
        output: list[ParsedNode],
    ) -> None:
        validate_node_payload(node, allow_children_missing=False)
        children = node.get("children", [])
        if not isinstance(children, list):
            raise TreeInsertionError(f"node at {jsonpath} has non-list children")

        children_summary = [
            ensure_str(child.get("node_name"))
            for child in children
            if isinstance(child, dict) and ensure_str(child.get("node_name"))
        ]
        children_texts = [
            build_node_text(child.get("node_name", ""), child.get("annotation", ""))
            for child in children
            if isinstance(child, dict)
        ]

        output.append(
            ParsedNode(
                node_id=ensure_str(node.get("node_id")),
                node_name=ensure_str(node.get("node_name")),
                node_type=ensure_str(node.get("node_type")),
                annotation=ensure_str(node.get("annotation")),
                jsonpath=jsonpath,
                depth=depth,
                ancestor_path_names=list(ancestor_names),
                ancestor_path_ids=list(ancestor_ids),
                children_summary=children_summary,
                children_texts=children_texts,
                raw_node=node,
            )
        )

        next_ancestor_names = ancestor_names + [ensure_str(node.get("node_name"))]
        next_ancestor_ids = ancestor_ids + [ensure_str(node.get("node_id"))]

        for index, child in enumerate(children):
            if not isinstance(child, dict):
                raise TreeInsertionError(f"child at {jsonpath}.children[{index}] must be a dictionary")
            child_path = self._build_child_path(jsonpath, index)
            self._walk(
                node=child,
                jsonpath=child_path,
                depth=depth + 1,
                ancestor_names=next_ancestor_names,
                ancestor_ids=next_ancestor_ids,
                output=output,
            )

    @staticmethod
    def _build_child_path(parent_path: str, child_index: int) -> str:
        return f"{parent_path}.children[{child_index}]"
