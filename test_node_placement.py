import copy
import unittest

from node_placement import PathError, PathParser, plan_nodes_by_json_path


class NodePlacementTest(unittest.TestCase):
    def setUp(self):
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

    def test_output_list_with_hierarchy_order(self):
        node = {
            "name": "Tax Rule",
            "annotation": "tax calculation logic",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0]",
        }
        result = plan_nodes_by_json_path(node, self.origin_tree, self.target_tree)

        self.assertFalse(result["is_exist"])
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(result["items"][0]["path"], "$.mapping_content")
        self.assertEqual(result["items"][0]["node"]["name"], "InvoiceInfo")
        self.assertEqual(result["items"][1]["path"], "$.mapping_content.children[0]")
        self.assertEqual(result["items"][1]["node"]["name"], "Tax Rule")
        self.assertEqual(result["working_tree"]["children"][0]["children"][0]["name"], "Tax Rule")

    def test_fallback_insert_path(self):
        node = {
            "name": "Fallback Child",
            "annotation": "should fallback",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0].children[0]",
        }
        result = plan_nodes_by_json_path(node, self.origin_tree, self.target_tree)

        self.assertFalse(result["is_exist"])
        self.assertEqual(result["items"][0]["path"], "$.mapping_content")
        self.assertEqual(result["items"][1]["path"], "$.mapping_content.children[0]")

    def test_duplicate_returns_exist_node_and_path(self):
        target_tree = copy.deepcopy(self.target_tree)
        target_tree["children"].append(
            {
                "name": "InvoiceInfo",
                "annotation": "invoice",
                "node_type": "parent",
                "children": [
                    {
                        "name": "Tax Rule",
                        "annotation": "dup",
                        "node_type": "simple_leaf",
                        "children": [],
                    }
                ],
            }
        )
        node = {
            "name": "Tax Rule",
            "annotation": "dup",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0]",
        }
        result = plan_nodes_by_json_path(node, self.origin_tree, target_tree)
        self.assertTrue(result["is_exist"])
        self.assertEqual(result["path"], "$.mapping_content.children[0].children[0]")
        self.assertEqual(result["node"]["name"], "Tax Rule")

    def test_path_invalid_insert_to_root(self):
        node = {
            "name": "Root Child",
            "annotation": "invalid path",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[9]",
        }
        result = plan_nodes_by_json_path(node, self.origin_tree, self.target_tree)
        self.assertFalse(result["is_exist"])
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["path"], "$.mapping_content")
        self.assertEqual(result["items"][0]["node"]["name"], "Root Child")

    def test_working_tree_supports_loop_calls_without_rebuilding_previous_nodes(self):
        first_node = {
            "name": "Tax Rule A",
            "annotation": "tax a",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0]",
        }
        first_result = plan_nodes_by_json_path(first_node, self.origin_tree, self.target_tree)

        second_node = {
            "name": "Tax Rule B",
            "annotation": "tax b",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0]",
        }
        second_result = plan_nodes_by_json_path(second_node, self.origin_tree, first_result["working_tree"])

        self.assertFalse(second_result["is_exist"])
        self.assertEqual(len(second_result["items"]), 1)
        self.assertEqual(second_result["items"][0]["path"], "$.mapping_content.children[0]")
        self.assertEqual(second_result["working_tree"]["children"][0]["children"][1]["name"], "Tax Rule B")


if __name__ == "__main__":
    unittest.main()
