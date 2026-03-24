from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any


INSERTABLE_NODE_TYPES = {
    "parent",
    "parent_list",
}

NON_INSERTABLE_NODE_TYPES = {
    "simple_leaf",
    "ab_pivot_table",
    "ab_two_level_table",
}


class InsertStatus(Enum):
    INSERTED = "inserted"
    DUPLICATE_SKIPPED = "duplicate_skipped"
    FALLBACK_INSERTED = "fallback_inserted"
    ROOT_INSERTED = "root_inserted"
    PATH_INVALID = "path_invalid"


class PathError(ValueError):
    """Raised when a json path cannot be resolved safely."""


@dataclass(frozen=True)
class ParsedPath:
    root_name: str
    child_indexes: list[int]


@dataclass
class BuildResult:
    candidate_parent_node: dict[str, Any]
    ancestor_chain: list[dict[str, Any]]


def _ensure_children(node: dict[str, Any]) -> list[dict[str, Any]]:
    children = node.get("children")
    if not isinstance(children, list):
        children = []
        node["children"] = children
    return children


def _matches_node(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a.get("name") == b.get("name")
        and a.get("annotation") == b.get("annotation")
        and a.get("node_type") == b.get("node_type")
    )


def _clone_node_metadata(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": node.get("name", ""),
        "annotation": node.get("annotation", ""),
        "node_type": node.get("node_type", ""),
        "children": [],
    }


def _node_display(node: dict[str, Any]) -> str:
    return f"{node.get('name', '')}[{node.get('node_type', '')}]"


class PathParser:
    _ROOT_PATTERN = re.compile(r"^\$\.([A-Za-z_][A-Za-z0-9_]*)")
    _CHILD_PATTERN = re.compile(r"\.children\[(\d+)\]")

    @classmethod
    def parse_json_path(cls, json_path: str) -> ParsedPath:
        if not isinstance(json_path, str) or not json_path:
            raise PathError("json_path must be a non-empty string")

        root_match = cls._ROOT_PATTERN.match(json_path)
        if not root_match:
            raise PathError(f"invalid json_path root: {json_path}")

        root_name = root_match.group(1)
        suffix = json_path[root_match.end() :]
        indexes: list[int] = []

        pos = 0
        while pos < len(suffix):
            child_match = cls._CHILD_PATTERN.match(suffix, pos)
            if not child_match:
                raise PathError(f"invalid children segment in json_path: {json_path}")
            indexes.append(int(child_match.group(1)))
            pos = child_match.end()

        return ParsedPath(root_name=root_name, child_indexes=indexes)


class ExistingTreeLocator:
    @staticmethod
    def locate_node_chain(existing_tree: dict[str, Any], parsed_path: ParsedPath) -> list[dict[str, Any]]:
        if existing_tree.get("name") != parsed_path.root_name:
            raise PathError(
                f"root name mismatch: path={parsed_path.root_name}, tree={existing_tree.get('name')}"
            )

        chain = [existing_tree]
        cursor = existing_tree

        for idx in parsed_path.child_indexes:
            children = cursor.get("children")
            if not isinstance(children, list) or idx < 0 or idx >= len(children):
                raise PathError(f"child index out of range: {idx}")
            child = children[idx]
            if not isinstance(child, dict):
                raise PathError("child node must be dict")
            chain.append(child)
            cursor = child

        return chain


class TreeBuilder:
    @staticmethod
    def build_path(current_tree: dict[str, Any], existing_node_chain: list[dict[str, Any]]) -> BuildResult:
        if not existing_node_chain:
            raise PathError("existing_node_chain cannot be empty")

        if not _matches_node(current_tree, existing_node_chain[0]):
            raise PathError("current_tree root and existing root are not aligned")

        cursor = current_tree
        ancestor_chain = [current_tree]

        for existing_node in existing_node_chain[1:]:
            children = _ensure_children(cursor)
            match = next((child for child in children if _matches_node(child, existing_node)), None)

            if match is None:
                match = _clone_node_metadata(existing_node)
                children.append(match)

            cursor = match
            ancestor_chain.append(cursor)

        return BuildResult(candidate_parent_node=cursor, ancestor_chain=ancestor_chain)


class InsertPositionResolver:
    @staticmethod
    def resolve_insert_position(
        candidate_parent_node: dict[str, Any],
        ancestor_chain: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], bool, bool]:
        """Return (final_parent_node, did_fallback, is_root)."""
        if candidate_parent_node.get("node_type") in INSERTABLE_NODE_TYPES:
            return candidate_parent_node, False, candidate_parent_node is ancestor_chain[0]

        for node in reversed(ancestor_chain):
            if node.get("node_type") in INSERTABLE_NODE_TYPES:
                return node, True, node is ancestor_chain[0]

        return ancestor_chain[0], True, True


class Deduplicator:
    @staticmethod
    def is_duplicate(parent_node: dict[str, Any], new_node: dict[str, Any]) -> bool:
        for child in _ensure_children(parent_node):
            if (
                child.get("name") == new_node.get("name")
                and child.get("annotation") == new_node.get("annotation")
            ):
                return True
        return False


class NodeInserter:
    @staticmethod
    def insert_node(parent_node: dict[str, Any], new_node: dict[str, Any]) -> None:
        _ensure_children(parent_node).append(new_node)


def _find_child_index(parent_node: dict[str, Any], child_node: dict[str, Any]) -> int:
    for i, child in enumerate(_ensure_children(parent_node)):
        if child is child_node:
            return i
    return -1


def _build_path_string(
    root_name: str,
    ancestor_chain: list[dict[str, Any]],
    final_parent_node: dict[str, Any],
) -> str:
    path = f"$.{root_name}"
    if final_parent_node is ancestor_chain[0]:
        return path

    current = ancestor_chain[0]
    for node in ancestor_chain[1:]:
        idx = _find_child_index(current, node)
        if idx < 0:
            return path
        path += f".children[{idx}]"
        current = node
        if node is final_parent_node:
            break
    return path


def place_node_by_json_path(
    new_node: dict[str, Any],
    json_path: str,
    existing_tree: dict[str, Any],
    current_tree: dict[str, Any],
) -> dict[str, Any]:
    root_name = existing_tree.get("name", "")

    try:
        parsed_path = PathParser.parse_json_path(json_path)
        node_chain = ExistingTreeLocator.locate_node_chain(existing_tree, parsed_path)
        build_result = TreeBuilder.build_path(current_tree, node_chain)
        final_parent, did_fallback, is_root = InsertPositionResolver.resolve_insert_position(
            build_result.candidate_parent_node,
            build_result.ancestor_chain,
        )

        final_path = _build_path_string(
            parsed_path.root_name,
            build_result.ancestor_chain,
            final_parent,
        )

        if Deduplicator.is_duplicate(final_parent, new_node):
            return {
                "updated_tree": current_tree,
                "insert_status": InsertStatus.DUPLICATE_SKIPPED,
                "final_insert_path": final_path,
            }

        NodeInserter.insert_node(final_parent, new_node)
        if did_fallback:
            status = InsertStatus.FALLBACK_INSERTED
        elif is_root:
            status = InsertStatus.ROOT_INSERTED
        else:
            status = InsertStatus.INSERTED

        return {
            "updated_tree": current_tree,
            "insert_status": status,
            "final_insert_path": final_path,
        }

    except PathError:
        if Deduplicator.is_duplicate(current_tree, new_node):
            return {
                "updated_tree": current_tree,
                "insert_status": InsertStatus.DUPLICATE_SKIPPED,
                "final_insert_path": f"$.{root_name}",
            }

        NodeInserter.insert_node(current_tree, new_node)
        return {
            "updated_tree": current_tree,
            "insert_status": InsertStatus.PATH_INVALID,
            "final_insert_path": f"$.{root_name}",
        }
