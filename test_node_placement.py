import copy
import unittest

from node_placement import InsertStatus, PathError, PathParser, plan_nodes_by_json_path


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

    def test_plan_nodes_with_missing_path_nodes(self):
        node = {
            "name": "Tax Rule",
            "annotation": "tax calculation logic",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0]",
        }
        result = plan_nodes_by_json_path(node, self.origin_tree, self.target_tree)

        self.assertEqual(result["insert_status"], InsertStatus.INSERTED)
        self.assertEqual(result["insert_json_path"], "$.mapping_content.children[0]")
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(result["nodes"][0]["name"], "InvoiceInfo")
        self.assertEqual(result["nodes"][1]["name"], "Tax Rule")

    def test_plan_nodes_fallback_insert(self):
        node = {
            "name": "Fallback Child",
            "annotation": "should fallback",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[0].children[0]",
        }
        result = plan_nodes_by_json_path(node, self.origin_tree, self.target_tree)

        self.assertEqual(result["insert_status"], InsertStatus.FALLBACK_INSERTED)
        self.assertEqual(result["insert_json_path"], "$.mapping_content.children[0]")
        self.assertEqual(result["nodes"][0]["name"], "InvoiceInfo")

    def test_plan_nodes_duplicate_skipped(self):
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

        self.assertEqual(result["insert_status"], InsertStatus.DUPLICATE_SKIPPED)
        self.assertEqual(result["insert_json_path"], "$.mapping_content.children[0]")
        self.assertEqual(result["nodes"], [])

    def test_plan_nodes_path_invalid(self):
        node = {
            "name": "Root Child",
            "annotation": "invalid path",
            "node_type": "simple_leaf",
            "children": [],
            "json_path": "$.mapping_content.children[9]",
        }
        result = plan_nodes_by_json_path(node, self.origin_tree, self.target_tree)

        self.assertEqual(result["insert_status"], InsertStatus.PATH_INVALID)
        self.assertEqual(result["insert_json_path"], "$.mapping_content")
        self.assertEqual(len(result["nodes"]), 1)
        self.assertEqual(result["nodes"][0]["name"], "Root Child")


if __name__ == "__main__":
    unittest.main()
