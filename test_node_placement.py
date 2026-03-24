import copy
import unittest

from node_placement import InsertStatus, PathParser, PathError, place_node_by_json_path


class NodePlacementTest(unittest.TestCase):
    def setUp(self):
        self.existing_tree = {
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
        self.base_current_tree = {
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

    def test_build_and_insert(self):
        current_tree = copy.deepcopy(self.base_current_tree)
        new_node = {
            "name": "Tax Rule",
            "annotation": "tax calculation logic",
            "node_type": "simple_leaf",
            "children": [],
        }
        result = place_node_by_json_path(
            new_node,
            "$.mapping_content.children[0]",
            self.existing_tree,
            current_tree,
        )
        self.assertEqual(result["insert_status"], InsertStatus.INSERTED)
        self.assertEqual(result["final_insert_path"], "$.mapping_content.children[0]")
        self.assertEqual(current_tree["children"][0]["name"], "InvoiceInfo")
        self.assertEqual(current_tree["children"][0]["children"][0]["name"], "Tax Rule")

    def test_fallback_insert(self):
        current_tree = copy.deepcopy(self.base_current_tree)
        new_node = {
            "name": "Fallback Child",
            "annotation": "should fallback",
            "node_type": "simple_leaf",
            "children": [],
        }
        result = place_node_by_json_path(
            new_node,
            "$.mapping_content.children[0].children[0]",
            self.existing_tree,
            current_tree,
        )
        self.assertEqual(result["insert_status"], InsertStatus.FALLBACK_INSERTED)
        self.assertEqual(result["final_insert_path"], "$.mapping_content.children[0]")
        self.assertEqual(len(current_tree["children"][0]["children"]), 2)

    def test_duplicate_skipped(self):
        current_tree = copy.deepcopy(self.base_current_tree)
        invoice = {
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
        current_tree["children"].append(invoice)
        new_node = {
            "name": "Tax Rule",
            "annotation": "dup",
            "node_type": "simple_leaf",
            "children": [],
        }
        result = place_node_by_json_path(
            new_node,
            "$.mapping_content.children[0]",
            self.existing_tree,
            current_tree,
        )
        self.assertEqual(result["insert_status"], InsertStatus.DUPLICATE_SKIPPED)
        self.assertEqual(len(current_tree["children"][0]["children"]), 1)

    def test_invalid_path_fallback_to_root(self):
        current_tree = copy.deepcopy(self.base_current_tree)
        new_node = {
            "name": "Root Child",
            "annotation": "invalid path",
            "node_type": "simple_leaf",
            "children": [],
        }
        result = place_node_by_json_path(
            new_node,
            "$.mapping_content.children[9]",
            self.existing_tree,
            current_tree,
        )
        self.assertEqual(result["insert_status"], InsertStatus.PATH_INVALID)
        self.assertEqual(result["final_insert_path"], "$.mapping_content")
        self.assertEqual(current_tree["children"][0]["name"], "Root Child")


if __name__ == "__main__":
    unittest.main()
