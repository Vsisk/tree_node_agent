from __future__ import annotations

import unittest

from tree_insertion_rag.candidate_builder import CandidateBuilder
from tree_insertion_rag.models import SearchAction
from tree_insertion_rag.retriever import TreeInsertionRetriever
from tree_insertion_rag.tree_parser import TreeParser


def sample_tree() -> dict:
    return {
        "mapping_content": {
            "node_name": "mapping_content",
            "node_id": "root",
            "node_type": "parent",
            "annotation": "票据结构映射根节点",
            "children": [
                {
                    "node_name": "基础信息",
                    "node_id": "p_basic",
                    "node_type": "parent",
                    "annotation": "发票基础抬头和编号信息",
                    "children": [
                        {
                            "node_name": "发票号码",
                            "node_id": "l_invoice_no",
                            "node_type": "leaf",
                            "annotation": "invoice number",
                        },
                        {
                            "node_name": "开票日期",
                            "node_id": "l_invoice_date",
                            "node_type": "leaf",
                            "annotation": "invoice issue date",
                        },
                    ],
                },
                {
                    "node_name": "费用明细",
                    "node_id": "p_fee",
                    "node_type": "parent",
                    "annotation": "金额、税费和收费项目容器",
                    "children": [
                        {
                            "node_name": "金额",
                            "node_id": "l_amount",
                            "node_type": "leaf",
                            "annotation": "总金额",
                        },
                        {
                            "node_name": "税额",
                            "node_id": "l_tax",
                            "node_type": "leaf",
                            "annotation": "税费金额",
                        },
                    ],
                },
                {
                    "node_name": "附录",
                    "node_id": "p_appendix",
                    "node_type": "parent",
                    "annotation": "备注和补充材料",
                    "children": [
                        {
                            "node_name": "备注",
                            "node_id": "l_remark",
                            "node_type": "leaf",
                            "annotation": "业务备注",
                        }
                    ],
                },
            ],
        }
    }


class TreeInsertionRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = TreeParser()
        self.builder = CandidateBuilder()
        self.retriever = TreeInsertionRetriever()

    def test_jsonpath_generation(self) -> None:
        parsed_nodes = self.parser.parse(sample_tree())
        path_map = {node.node_id: node.jsonpath for node in parsed_nodes}
        self.assertEqual(path_map["root"], "$.mapping_content")
        self.assertEqual(path_map["p_basic"], "$.mapping_content.children[0]")
        self.assertEqual(path_map["l_tax"], "$.mapping_content.children[1].children[1]")

    def test_candidates_only_include_parent_nodes(self) -> None:
        parsed_nodes = self.parser.parse(sample_tree())
        candidates = self.builder.build(parsed_nodes)
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.node_type == "parent" for candidate in candidates))
        self.assertNotIn("l_tax", {candidate.node_id for candidate in candidates})

    def test_modify_candidates_include_leaf_nodes(self) -> None:
        parsed_nodes = self.parser.parse(sample_tree())
        candidates = self.builder.build(parsed_nodes, action=SearchAction.MODIFY)
        self.assertIn("l_tax", {candidate.node_id for candidate in candidates})

    def test_leaf_node_will_not_be_returned(self) -> None:
        result = self.retriever.find_best_node(
            tree=sample_tree(),
            node={
                "node_name": "服务费",
                "node_id": "n_service_fee",
                "node_type": "leaf",
                "annotation": "订单服务费金额",
            },
            query="该字段属于费用明细，和金额、税额同级。",
            action="add",
            topk=5,
        )
        self.assertEqual(result["jsonpath"], "$.mapping_content.children[1]")
        self.assertNotEqual(result["jsonpath"], "$.mapping_content.children[1].children[0]")
        self.assertEqual(result["matched_node_id"], "p_fee")

    def test_query_hits_expected_parent(self) -> None:
        result = self.retriever.find_best_node(
            tree=sample_tree(),
            node={
                "node_name": "税率",
                "node_id": "n_tax_rate",
                "node_type": "leaf",
                "annotation": "税费比例",
            },
            query="这个字段属于费用明细模块，和金额、税额一组，表示税费相关信息。",
            action="add",
            topk=5,
        )
        self.assertEqual(result["jsonpath"], "$.mapping_content.children[1]")
        self.assertIn(result["confidence"], {"medium", "high"})

    def test_add_low_confidence_returns_fallback_path(self) -> None:
        result = self.retriever.find_best_node(
            tree=sample_tree(),
            node={
                "node_name": "日志级别",
                "node_id": "n_log_level",
                "node_type": "leaf",
                "annotation": "系统运行时调试开关",
            },
            query="这是一个和当前票据结构无关的系统日志配置项。",
            action="add",
            topk=5,
        )
        self.assertIsNotNone(result["jsonpath"])
        self.assertIsNotNone(result["matched_node_id"])
        self.assertEqual(result["confidence"], "low")

    def test_modify_can_hit_leaf_node(self) -> None:
        result = self.retriever.find_best_node(
            tree=sample_tree(),
            node={
                "node_name": "税额",
                "node_id": "n_new_tax_alias",
                "node_type": "leaf",
                "annotation": "税费金额字段",
            },
            query="请修改税额字段的名称和说明。",
            action="modify",
            topk=5,
        )
        self.assertEqual(result["jsonpath"], "$.mapping_content.children[1].children[1]")
        self.assertEqual(result["matched_node_id"], "l_tax")

    def test_delete_can_work_without_node_payload(self) -> None:
        result = self.retriever.find_best_node(
            tree=sample_tree(),
            query="删除附录下面的备注节点。",
            action="delete",
            topk=5,
        )
        self.assertEqual(result["jsonpath"], "$.mapping_content.children[2].children[0]")
        self.assertEqual(result["matched_node_id"], "l_remark")


if __name__ == "__main__":
    unittest.main()
