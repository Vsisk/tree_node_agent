from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

from jsonpath_lib import FieldToken, IndexToken, JsonPathSyntaxError, parse as parse_jsonpath


INSERTABLE_NODE_TYPES = {"parent", "parent_list"}
NON_INSERTABLE_NODE_TYPES = {"simple_leaf", "ab_pivot_table", "ab_two_level_table"}


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
    ancestor_paths: list[str]
    created_items: list[dict[str, Any]]


def _ensure_children(node: dict[str, Any]) -> list[dict[str, Any]]:
    children = node.get("children")
    if not isinstance(children, list):
        node["children"] = []
    return node["children"]


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


def _find_child_index(parent_node: dict[str, Any], child_node: dict[str, Any]) -> int:
    for i, child in enumerate(_ensure_children(parent_node)):
        if child is child_node:
            return i
    return -1


class PathParser:
    @classmethod
    def parse_json_path(cls, json_path: str) -> ParsedPath:
        try:
            parsed = parse_jsonpath(json_path)
        except JsonPathSyntaxError as exc:
            raise PathError(str(exc)) from exc

        if not parsed.tokens:
            raise PathError("json_path must contain root field")

        first_token = parsed.tokens[0]
        if not isinstance(first_token, FieldToken):
            raise PathError("json_path must start with root field after '$'")

        root_name = first_token.name
        child_indexes: list[int] = []

        i = 1
        tokens = parsed.tokens
        while i < len(tokens):
            token = tokens[i]
            if not isinstance(token, FieldToken):
                raise PathError("invalid token order in json_path")
            if token.name != "children":
                raise PathError(f"unsupported field in json_path: {token.name}")
            i += 1
            if i >= len(tokens) or not isinstance(tokens[i], IndexToken):
                raise PathError("children field must be followed by index")
            child_indexes.append(tokens[i].value)
            i += 1

        return ParsedPath(root_name=root_name, child_indexes=child_indexes)


class ExistingTreeLocator:
    @staticmethod
    def locate_node_chain(existing_tree: dict[str, Any], parsed_path: ParsedPath) -> list[dict[str, Any]]:
        if existing_tree.get("name") != parsed_path.root_name:
            raise PathError(
                f"root name mismatch: path={parsed_path.root_name}, tree={existing_tree.get('name')}"
            )

        node_chain = [existing_tree]
        cursor = existing_tree
        for idx in parsed_path.child_indexes:
            children = cursor.get("children")
            if not isinstance(children, list) or idx < 0 or idx >= len(children):
                raise PathError(f"child index out of range: {idx}")
            child = children[idx]
            if not isinstance(child, dict):
                raise PathError("child node must be dict")
            node_chain.append(child)
            cursor = child
        return node_chain


class TreeBuilder:
    @staticmethod
    def build_path(current_tree: dict[str, Any], existing_node_chain: list[dict[str, Any]]) -> BuildResult:
        if not existing_node_chain:
            raise PathError("existing_node_chain cannot be empty")
        if not _matches_node(current_tree, existing_node_chain[0]):
            raise PathError("current_tree root and existing root are not aligned")

        root_path = f"$.{existing_node_chain[0].get('name', '')}"
        cursor = current_tree
        cursor_path = root_path

        ancestor_chain = [current_tree]
        ancestor_paths = [root_path]
        created_items: list[dict[str, Any]] = []

        for existing_node in existing_node_chain[1:]:
            children = _ensure_children(cursor)
            match = next((child for child in children if _matches_node(child, existing_node)), None)
            if match is None:
                match = _clone_node_metadata(existing_node)
                children.append(match)
                child_index = len(children) - 1
                match_path = f"{cursor_path}.children[{child_index}]"
                created_items.append({"path": match_path, "node": match})
            else:
                child_index = _find_child_index(cursor, match)
                match_path = f"{cursor_path}.children[{child_index}]"

            cursor = match
            cursor_path = match_path
            ancestor_chain.append(cursor)
            ancestor_paths.append(cursor_path)

        return BuildResult(
            candidate_parent_node=cursor,
            ancestor_chain=ancestor_chain,
            ancestor_paths=ancestor_paths,
            created_items=created_items,
        )


class InsertPositionResolver:
    @staticmethod
    def resolve_insert_position(
        candidate_parent_node: dict[str, Any],
        ancestor_chain: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], bool, bool]:
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
            if child.get("name") == new_node.get("name") and child.get("annotation") == new_node.get("annotation"):
                return True
        return False


def _build_insert_item(parent_path: str, parent_node: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    insert_index = len(_ensure_children(parent_node))
    return {
        "path": f"{parent_path}.children[{insert_index}]",
        "node": node,
    }


def plan_nodes_by_json_path(
    node: dict[str, Any],
    origin_tree: dict[str, Any],
    target_tree: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    输出新增节点列表，格式：[{"path": "$.a.children[0]", "node": {...}}, ...]。
    顺序保证按层级从上到下。
    """
    working_tree = deepcopy(target_tree)
    raw_json_path = node.get("json_path", "")
    payload_node = {k: deepcopy(v) for k, v in node.items() if k != "json_path"}

    try:
        parsed_path = PathParser.parse_json_path(raw_json_path)
        existing_chain = ExistingTreeLocator.locate_node_chain(origin_tree, parsed_path)
        build_result = TreeBuilder.build_path(working_tree, existing_chain)

        final_parent, _, _ = InsertPositionResolver.resolve_insert_position(
            build_result.candidate_parent_node,
            build_result.ancestor_chain,
        )
        parent_index = next(
            (i for i, ancestor in enumerate(build_result.ancestor_chain) if ancestor is final_parent),
            0,
        )
        final_parent_path = build_result.ancestor_paths[parent_index]

        if Deduplicator.is_duplicate(final_parent, payload_node):
            return build_result.created_items

        return [
            *build_result.created_items,
            _build_insert_item(final_parent_path, final_parent, payload_node),
        ]

    except PathError:
        root_path = f"$.{origin_tree.get('name', '')}"
        if Deduplicator.is_duplicate(working_tree, payload_node):
            return []

        return [_build_insert_item(root_path, working_tree, payload_node)]
