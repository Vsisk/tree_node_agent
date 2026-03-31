from __future__ import annotations

import json
import logging

from tree_insertion_rag.retriever import TreeInsertionRetriever


def build_demo_tree() -> dict:
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
                        {
                            "node_name": "币种",
                            "node_id": "l_currency",
                            "node_type": "leaf",
                            "annotation": "交易币种",
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


def build_demo_target() -> dict:
    return {
        "node_name": "服务费",
        "node_id": "n_205",
        "node_type": "leaf",
        "annotation": "订单服务费金额",
    }


def build_demo_query() -> str:
    return "这个字段属于费用明细，和金额、税额同级，表示收费项目中的服务费金额。"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    retriever = TreeInsertionRetriever(verbose=True)
    result = retriever.find_best_node(
        tree=build_demo_tree(),
        query=build_demo_query(),
        action="add",
        node=build_demo_target(),
        topk=5,
        verbose=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
