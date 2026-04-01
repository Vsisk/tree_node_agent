from __future__ import annotations

import unittest

from tree_insertion_rag.parser import TreeParser
from tree_insertion_rag.ranker import Ranker
from tree_insertion_rag.selector import TreeInsertionSelector


class StubEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vocabulary = ["basic", "fee", "appendix", "remark", "tax", "amount", "invoice", "service", "date"]
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append([float(lowered.count(token)) for token in vocabulary])
        return vectors


def sample_tree() -> dict:
    return {
        "mapping_content": {
            "node_name": "mapping_content",
            "node_id": "root",
            "node_type": "parent",
            "annotation": "invoice mapping root",
            "children": [
                {
                    "node_name": "basic_info",
                    "node_id": "p_basic",
                    "node_type": "parent",
                    "annotation": "invoice header information",
                    "children": [
                        {
                            "node_name": "invoice_no",
                            "node_id": "l_invoice_no",
                            "node_type": "leaf",
                            "annotation": "invoice number",
                        },
                        {
                            "node_name": "invoice_date",
                            "node_id": "l_invoice_date",
                            "node_type": "leaf",
                            "annotation": "invoice issue date",
                        },
                    ],
                },
                {
                    "node_name": "fee_detail",
                    "node_id": "p_fee",
                    "node_type": "parent",
                    "annotation": "amount and tax fields",
                    "children": [
                        {
                            "node_name": "amount",
                            "node_id": "l_amount",
                            "node_type": "leaf",
                            "annotation": "total amount",
                        },
                        {
                            "node_name": "tax",
                            "node_id": "l_tax",
                            "node_type": "leaf",
                            "annotation": "tax amount",
                        },
                    ],
                },
                {
                    "node_name": "appendix",
                    "node_id": "p_appendix",
                    "node_type": "parent",
                    "annotation": "remarks and attachments",
                    "children": [
                        {
                            "node_name": "remark",
                            "node_id": "l_remark",
                            "node_type": "leaf",
                            "annotation": "business remark",
                        }
                    ],
                },
            ],
        }
    }


class TreeInsertionSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = TreeParser()
        self.ranker = Ranker(embedding_model=StubEmbeddingModel())
        self.selector = TreeInsertionSelector(ranker=self.ranker)

    def test_jsonpath_generation(self) -> None:
        parsed_nodes = self.parser.parse(sample_tree())
        path_map = {node.node_id: node.jsonpath for node in parsed_nodes}
        self.assertEqual(path_map["root"], "$.mapping_content")
        self.assertEqual(path_map["p_basic"], "$.mapping_content.children[0]")
        self.assertEqual(path_map["l_tax"], "$.mapping_content.children[1].children[1]")

    def test_add_only_ranks_parent_candidates(self) -> None:
        ranked = self.ranker.rank(
            parsed_nodes=self.parser.parse(sample_tree()),
            query="service fee belongs with amount and tax",
            action="add",
            target_node={
                "node_name": "service_fee",
                "node_id": "n_service_fee",
                "node_type": "leaf",
                "annotation": "service fee amount",
            },
            topk=5,
        )
        self.assertTrue(ranked)
        self.assertTrue(all(item.candidate.node_type == "parent" for item in ranked))

    def test_add_returns_expected_parent_jsonpath(self) -> None:
        result = self.selector.find_best_node(
            tree=sample_tree(),
            node={
                "node_name": "service_fee",
                "node_id": "n_service_fee",
                "node_type": "leaf",
                "annotation": "service fee amount",
            },
            query="service fee belongs with amount and tax",
            action="add",
            topk=5,
        )
        self.assertEqual(result, "$.mapping_content.children[1]")

    def test_modify_can_return_leaf_jsonpath(self) -> None:
        result = self.selector.find_best_node(
            tree=sample_tree(),
            node={
                "node_name": "tax",
                "node_id": "n_tax_alias",
                "node_type": "leaf",
                "annotation": "tax amount field",
            },
            query="modify the tax field",
            action="modify",
            topk=5,
        )
        self.assertEqual(result, "$.mapping_content.children[1].children[1]")

    def test_delete_can_work_without_node_payload(self) -> None:
        result = self.selector.find_best_node(
            tree=sample_tree(),
            query="delete the remark under appendix",
            action="delete",
            topk=5,
        )
        self.assertEqual(result, "$.mapping_content.children[2].children[0]")

    def test_llm_selector_can_override_ranker_top1(self) -> None:
        class StubLLMSelector:
            def select(self, query, action, target_node, ranked_candidates):
                del query, action, target_node
                return next(item for item in ranked_candidates if item.candidate.node_id == "p_appendix")

        selector = TreeInsertionSelector(
            ranker=Ranker(embedding_model=StubEmbeddingModel()),
            candidate_selector=StubLLMSelector(),
        )
        result = selector.find_best_node(
            tree=sample_tree(),
            node={
                "node_name": "attachment_desc",
                "node_id": "n_attachment_desc",
                "node_type": "leaf",
                "annotation": "attachment description",
            },
            query="attachment details should go to appendix",
            action="add",
            topk=5,
        )
        self.assertEqual(result, "$.mapping_content.children[2]")


if __name__ == "__main__":
    unittest.main()
