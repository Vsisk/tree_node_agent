from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REQUIRED_NODE_FIELDS = {"node_name", "node_id", "node_type", "annotation"}


class TreeInsertionError(ValueError):
    """Raised when the tree structure or input payload is invalid."""


@dataclass(slots=True)
class ParsedNode:
    node_id: str
    node_name: str
    node_type: str
    annotation: str
    jsonpath: str
    depth: int
    ancestor_names: list[str] = field(default_factory=list)
    children_texts: list[str] = field(default_factory=list)
    raw_node: dict[str, Any] = field(default_factory=dict)


def ensure_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def build_node_text(node_name: str, annotation: str) -> str:
    parts = [ensure_str(node_name), ensure_str(annotation)]
    return " ".join(part for part in parts if part)


def validate_node(node: dict[str, Any], *, require_children: bool) -> None:
    if not isinstance(node, dict):
        raise TreeInsertionError("node must be a dictionary")
    missing = [field for field in REQUIRED_NODE_FIELDS if field not in node]
    if missing:
        raise TreeInsertionError(f"node is missing required fields: {', '.join(sorted(missing))}")
    if require_children and not isinstance(node.get("children"), list):
        raise TreeInsertionError("parent node must contain a list field named 'children'")


class TreeParser:
    """Parse the tree into flattened nodes with stable jsonpath metadata."""

    def parse(self, tree: dict[str, Any]) -> list[ParsedNode]:
        root_node, root_path = self._resolve_root(tree)
        output: list[ParsedNode] = []
        self._walk(
            node=root_node,
            jsonpath=root_path,
            depth=0,
            ancestor_names=[],
            output=output,
        )
        return output

    def _resolve_root(self, tree: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if self._is_node(tree):
            validate_node(tree, require_children=tree.get("node_type") == "parent")
            return tree, "$"

        if not isinstance(tree, dict) or len(tree) != 1:
            raise TreeInsertionError("tree must be a wrapped root object like {'mapping_content': {...}}")

        root_key, root_value = next(iter(tree.items()))
        if not self._is_node(root_value):
            raise TreeInsertionError("wrapped root must be a valid node object")
        validate_node(root_value, require_children=root_value.get("node_type") == "parent")
        return root_value, f"$.{root_key}"

    def _walk(
        self,
        node: dict[str, Any],
        jsonpath: str,
        depth: int,
        ancestor_names: list[str],
        output: list[ParsedNode],
    ) -> None:
        validate_node(node, require_children=node.get("node_type") == "parent")
        children = node.get("children", [])
        if not isinstance(children, list):
            raise TreeInsertionError(f"node at {jsonpath} has non-list children")

        output.append(
            ParsedNode(
                node_id=ensure_str(node.get("node_id")),
                node_name=ensure_str(node.get("node_name")),
                node_type=ensure_str(node.get("node_type")),
                annotation=ensure_str(node.get("annotation")),
                jsonpath=jsonpath,
                depth=depth,
                ancestor_names=list(ancestor_names),
                children_texts=[
                    build_node_text(child.get("node_name", ""), child.get("annotation", ""))
                    for child in children
                    if isinstance(child, dict)
                ],
                raw_node=node,
            )
        )

        next_ancestors = ancestor_names + [ensure_str(node.get("node_name"))]
        for index, child in enumerate(children):
            if not isinstance(child, dict):
                raise TreeInsertionError(f"child at {jsonpath}.children[{index}] must be a dictionary")
            self._walk(
                node=child,
                jsonpath=f"{jsonpath}.children[{index}]",
                depth=depth + 1,
                ancestor_names=next_ancestors,
                output=output,
            )

    @staticmethod
    def _is_node(value: Any) -> bool:
        return isinstance(value, dict) and REQUIRED_NODE_FIELDS.issubset(value.keys())
