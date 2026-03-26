import copy
import unittest

from node_placement import (
    CommitStatus,
    PathError,
    PathParser,
    TREE_STATE_MANAGER,
    plan_nodes_by_json_path,
)


class NodePlacementTest(unittest.TestCase):
    def setUp(self):
        with TREE_STATE_MANAGER._lock:
            TREE_STATE_MANAGER._working_tree = None
            TREE_STATE_MANAGER._version = 0

        self.origin_tree = {
            "name": "mapping_content",
            "annotation": "root",
            "node_type": "parent",
            "children": [
                {
                    "name": "InvoiceInfo",
                    "annotation": "invoice",
                    "node_type": "parent",
                    "children": [
                        {
                            "name": "TaxInfo",
                            "annotation": "tax",
                            "node_type": "simple_leaf",
                            "children": [],
                        }
                    ],
                }
            ],
        }
        self.target_tree = {
            "name": "mapping_content",
            "annotation": "root",
            "node_type": "parent",
            "children": [],
        }

    def test_path_parser(self):
        parsed = PathParser.parse_json_path("$.mapping_content.children[0].children[1]")
        self.assertEqual(parsed.root_name, "mapping_content")
        self.assertEqual(parsed.child_indexes, [0, 1])

    def test_path_parser_invalid(self):
        with self.assertRaises(PathError):
            PathParser.parse_json_path("$.mapping_content.bad[0]")

    def test_single_thread_success(self):
        node = {
            "name": "Tax Rule",
            "annotation": "tax calculation logic",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0]",
        }
        result = plan_nodes_by_json_path(node, self.origin_tree, self.target_tree)

        self.assertFalse(result["is_exist"])
        self.assertEqual(result["commit_status"], CommitStatus.SUCCESS.value)
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(result["tree_version"], 1)

    def test_repositioned_and_success_when_version_changed(self):
        node_a = {
            "name": "A",
            "annotation": "a",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0]",
        }
        node_b = {
            "name": "B",
            "annotation": "b",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0]",
        }

        first = plan_nodes_by_json_path(node_a, self.origin_tree, self.target_tree)
        self.assertEqual(first["commit_status"], CommitStatus.SUCCESS.value)

        stale_target = copy.deepcopy(self.target_tree)
        second = plan_nodes_by_json_path(node_b, self.origin_tree, stale_target)
        self.assertEqual(second["commit_status"], CommitStatus.REPOSITIONED_AND_SUCCESS.value)

    def test_conflict_when_reposition_failed(self):
        node_a = {
            "name": "A",
            "annotation": "a",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0]",
        }
        first = plan_nodes_by_json_path(node_a, self.origin_tree, self.target_tree)
        self.assertEqual(first["commit_status"], CommitStatus.SUCCESS.value)

        # 模拟并发方把目标父节点类型改坏
        with TREE_STATE_MANAGER._lock:
            TREE_STATE_MANAGER._working_tree["children"][0]["node_type"] = "simple_leaf"

        node_b = {
            "name": "B",
            "annotation": "b",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0]",
        }
        second = plan_nodes_by_json_path(node_b, self.origin_tree, self.target_tree)
        self.assertEqual(second["commit_status"], CommitStatus.CONFLICT.value)

    def test_invalid_parent_type(self):
        bad_origin = copy.deepcopy(self.origin_tree)
        bad_origin["children"][0]["node_type"] = "simple_leaf"

        node = {
            "name": "C",
            "annotation": "c",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0]",
        }
        result = plan_nodes_by_json_path(node, bad_origin, self.target_tree)
        self.assertEqual(result["commit_status"], CommitStatus.INVALID_PARENT_TYPE.value)

    def test_invalid_path(self):
        node = {
            "name": "Root Child",
            "annotation": "invalid path",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[9]",
        }
        result = plan_nodes_by_json_path(node, self.origin_tree, self.target_tree)
        self.assertEqual(result["commit_status"], CommitStatus.INVALID_PATH.value)


if __name__ == "__main__":
    unittest.main()
