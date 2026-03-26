from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import logging
import threading
from typing import Any

from jsonpath_lib import FieldToken, IndexToken, JsonPathSyntaxError, parse as parse_jsonpath


logger = logging.getLogger(__name__)

INSERTABLE_NODE_TYPES = {"parent", "parent_list"}
NON_INSERTABLE_NODE_TYPES = {"simple_leaf", "ab_pivot_table", "ab_two_level_table"}


class InsertStatus(Enum):
    INSERTED = "inserted"
    DUPLICATE_SKIPPED = "duplicate_skipped"
    FALLBACK_INSERTED = "fallback_inserted"
    ROOT_INSERTED = "root_inserted"
    PATH_INVALID = "path_invalid"


class PlanStatus(Enum):
    OK = "ok"
    INVALID_PATH = "invalid_path"
    INVALID_PARENT_TYPE = "invalid_parent_type"
    PLAN_FAILED = "plan_failed"


class CommitStatus(Enum):
    SUCCESS = "success"
    REPOSITIONED_AND_SUCCESS = "repositioned_and_success"
    CONFLICT = "conflict"
    INVALID_PATH = "invalid_path"
    INVALID_PARENT_TYPE = "invalid_parent_type"
    INSERT_FAILED = "insert_failed"


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


@dataclass
class InsertPlan:
    base_version: int
    target_json_path: str
    resolved_parent_path: str | None
    insert_index: int | None
    new_node: dict[str, Any]
    validation_summary: list[dict[str, Any]]
    status: PlanStatus
    message: str


@dataclass
class CommitResult:
    status: CommitStatus
    message: str
    is_repositioned: bool
    is_exist: bool
    existing_path: str | None
    existing_node: dict[str, Any] | None
    items: list[dict[str, Any]]
    working_tree: dict[str, Any]
    tree_version: int


class TreeStateManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._working_tree: dict[str, Any] | None = None
        self._version = 0

    def get_snapshot(self, target_tree: dict[str, Any]) -> tuple[dict[str, Any], int]:
        with self._lock:
            if self._working_tree is None:
                self._working_tree = deepcopy(target_tree)
                self._version = 0
            # 分析阶段基于调用方提供的快照执行，不直接读取最新树，避免隐藏并发漂移
            return deepcopy(target_tree), self._version

    def read_current_in_lock(self) -> tuple[dict[str, Any], int]:
        if self._working_tree is None:
            raise RuntimeError("Tree state is not initialized")
        return self._working_tree, self._version

    def set_current_in_lock(self, updated_tree: dict[str, Any]) -> int:
        self._working_tree = updated_tree
        self._version += 1
        return self._version


TREE_STATE_MANAGER = TreeStateManager()


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


def _build_child_path(parent_path: str, child_index: int) -> str:
    return f"{parent_path}.children[{child_index}]"


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
                match_path = _build_child_path(cursor_path, child_index)
                created_items.append({"path": cursor_path, "node": match})
            else:
                child_index = _find_child_index(cursor, match)
                match_path = _build_child_path(cursor_path, child_index)

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
    def find_duplicate(parent_node: dict[str, Any], new_node: dict[str, Any]) -> tuple[dict[str, Any], int] | None:
        for idx, child in enumerate(_ensure_children(parent_node)):
            if child.get("name") == new_node.get("name") and child.get("annotation") == new_node.get("annotation"):
                return child, idx
        return None


def _build_insert_item(parent_path: str, parent_node: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    _ensure_children(parent_node)
    return {"path": parent_path, "node": node}


def _resolve_target_parent(
    tree: dict[str, Any], origin_tree: dict[str, Any], json_path: str
) -> tuple[BuildResult, dict[str, Any], str, list[dict[str, Any]]]:
    parsed_path = PathParser.parse_json_path(json_path)
    existing_chain = ExistingTreeLocator.locate_node_chain(origin_tree, parsed_path)

    cursor = tree
    for depth, idx in enumerate(parsed_path.child_indexes):
        children = _ensure_children(cursor)
        if idx < len(children):
            actual = children[idx]
            expected = existing_chain[depth + 1]
            if not _matches_node(actual, expected):
                raise PathError("path occupied by different node")
            cursor = actual
        else:
            break

    build_result = TreeBuilder.build_path(tree, existing_chain)

    final_parent = build_result.candidate_parent_node
    parent_index = len(build_result.ancestor_chain) - 1
    final_parent_path = build_result.ancestor_paths[parent_index]
    validation_summary = [
        {
            "path": build_result.ancestor_paths[i],
            "name": node.get("name"),
            "annotation": node.get("annotation"),
            "node_type": node.get("node_type"),
        }
        for i, node in enumerate(build_result.ancestor_chain[: parent_index + 1])
    ]
    return build_result, final_parent, final_parent_path, validation_summary


def _validate_summary_against_tree(tree: dict[str, Any], summary: list[dict[str, Any]]) -> bool:
    if not summary:
        return False
    try:
        parsed = PathParser.parse_json_path(summary[-1]["path"])
        chain = ExistingTreeLocator.locate_node_chain(tree, parsed)
    except Exception:
        return False

    if len(chain) != len(summary):
        return False

    for snap, node in zip(summary, chain):
        if (
            snap.get("name") != node.get("name")
            or snap.get("annotation") != node.get("annotation")
            or snap.get("node_type") != node.get("node_type")
        ):
            return False
    return True


def build_insert_plan(
    snapshot_tree: dict[str, Any],
    base_version: int,
    json_path: str,
    new_node: dict[str, Any],
    origin_tree: dict[str, Any],
) -> InsertPlan:
    try:
        _, final_parent, final_parent_path, summary = _resolve_target_parent(
            snapshot_tree,
            origin_tree,
            json_path,
        )
    except PathError as exc:
        return InsertPlan(
            base_version=base_version,
            target_json_path=json_path,
            resolved_parent_path=None,
            insert_index=None,
            new_node=deepcopy(new_node),
            validation_summary=[],
            status=PlanStatus.INVALID_PATH,
            message=str(exc),
        )

    if final_parent.get("node_type") not in INSERTABLE_NODE_TYPES:
        return InsertPlan(
            base_version=base_version,
            target_json_path=json_path,
            resolved_parent_path=final_parent_path,
            insert_index=None,
            new_node=deepcopy(new_node),
            validation_summary=summary,
            status=PlanStatus.INVALID_PARENT_TYPE,
            message=f"invalid parent type: {final_parent.get('node_type')}",
        )

    insert_index = len(_ensure_children(final_parent))
    return InsertPlan(
        base_version=base_version,
        target_json_path=json_path,
        resolved_parent_path=final_parent_path,
        insert_index=insert_index,
        new_node=deepcopy(new_node),
        validation_summary=summary,
        status=PlanStatus.OK,
        message="ok",
    )


def commit_insert_plan(plan: InsertPlan, origin_tree: dict[str, Any]) -> CommitResult:
    with TREE_STATE_MANAGER._lock:
        current_tree, current_version = TREE_STATE_MANAGER.read_current_in_lock()

        logger.info(
            "commit start base_version=%s current_version=%s status=%s",
            plan.base_version,
            current_version,
            plan.status.value,
        )

        if plan.status == PlanStatus.INVALID_PATH:
            return CommitResult(
                status=CommitStatus.INVALID_PATH,
                message=plan.message,
                is_repositioned=False,
                is_exist=False,
                existing_path=None,
                existing_node=None,
                items=[],
                working_tree=deepcopy(current_tree),
                tree_version=current_version,
            )
        if plan.status == PlanStatus.INVALID_PARENT_TYPE:
            return CommitResult(
                status=CommitStatus.INVALID_PARENT_TYPE,
                message=plan.message,
                is_repositioned=False,
                is_exist=False,
                existing_path=None,
                existing_node=None,
                items=[],
                working_tree=deepcopy(current_tree),
                tree_version=current_version,
            )

        should_reposition = current_version != plan.base_version
        if not should_reposition and plan.resolved_parent_path:
            try:
                chain = ExistingTreeLocator.locate_node_chain(
                    current_tree,
                    PathParser.parse_json_path(plan.resolved_parent_path),
                )
                if plan.insert_index is not None and len(_ensure_children(chain[-1])) != plan.insert_index:
                    should_reposition = True
                elif not _validate_summary_against_tree(current_tree, plan.validation_summary):
                    should_reposition = True
            except PathError:
                # parent path may not exist yet because it will be created during commit
                should_reposition = False

        active_plan = plan
        repositioned = False
        if should_reposition:
            repositioned = True
            logger.info("reposition triggered base=%s current=%s", plan.base_version, current_version)
            rebuilt = build_insert_plan(
                deepcopy(current_tree),
                current_version,
                plan.target_json_path,
                plan.new_node,
                origin_tree,
            )
            active_plan = rebuilt
            if rebuilt.status != PlanStatus.OK:
                return CommitResult(
                    status=CommitStatus.CONFLICT,
                    message=f"reposition failed: {rebuilt.message}",
                    is_repositioned=True,
                    is_exist=False,
                    existing_path=None,
                    existing_node=None,
                    items=[],
                    working_tree=deepcopy(current_tree),
                    tree_version=current_version,
                )

        try:
            build_result, final_parent, final_parent_path, _ = _resolve_target_parent(
                current_tree,
                origin_tree,
                active_plan.target_json_path,
            )
        except Exception as exc:
            return CommitResult(
                status=CommitStatus.INSERT_FAILED,
                message=str(exc),
                is_repositioned=repositioned,
                is_exist=False,
                existing_path=None,
                existing_node=None,
                items=[],
                working_tree=deepcopy(current_tree),
                tree_version=current_version,
            )

        duplicate = Deduplicator.find_duplicate(final_parent, active_plan.new_node)
        if duplicate is not None:
            dup_node, dup_idx = duplicate
            return CommitResult(
                status=CommitStatus.SUCCESS,
                message="node already exists",
                is_repositioned=repositioned,
                is_exist=True,
                existing_path=_build_child_path(final_parent_path, dup_idx),
                existing_node=dup_node,
                items=[],
                working_tree=deepcopy(current_tree),
                tree_version=current_version,
            )

        insert_index = len(_ensure_children(final_parent))
        _ensure_children(final_parent).append(deepcopy(active_plan.new_node))

        new_version = TREE_STATE_MANAGER.set_current_in_lock(current_tree)
        items = [*build_result.created_items, _build_insert_item(final_parent_path, final_parent, active_plan.new_node)]

        return CommitResult(
            status=CommitStatus.REPOSITIONED_AND_SUCCESS if repositioned else CommitStatus.SUCCESS,
            message="inserted",
            is_repositioned=repositioned,
            is_exist=False,
            existing_path=None,
            existing_node=None,
            items=items,
            working_tree=deepcopy(current_tree),
            tree_version=new_version,
        )


def plan_nodes_by_json_path(
    node: dict[str, Any],
    origin_tree: dict[str, Any],
    target_tree: dict[str, Any],
) -> dict[str, Any]:
    raw_json_path = node.get("json_path", "")
    payload_node = {k: deepcopy(v) for k, v in node.items() if k != "json_path"}

    snapshot_tree, base_version = TREE_STATE_MANAGER.get_snapshot(target_tree)
    plan = build_insert_plan(snapshot_tree, base_version, raw_json_path, payload_node, origin_tree)
    logger.info(
        "plan built base_version=%s target=%s status=%s",
        base_version,
        raw_json_path,
        plan.status.value,
    )

    commit = commit_insert_plan(plan, origin_tree)

    if commit.is_exist:
        return {
            "is_exist": True,
            "path": commit.existing_path,
            "node": commit.existing_node,
            "working_tree": commit.working_tree,
            "commit_status": commit.status.value,
            "message": commit.message,
            "tree_version": commit.tree_version,
        }

    return {
        "is_exist": False,
        "items": commit.items,
        "working_tree": commit.working_tree,
        "commit_status": commit.status.value,
        "message": commit.message,
        "tree_version": commit.tree_version,
    }
