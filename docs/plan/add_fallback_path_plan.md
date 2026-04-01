# Add 任务低置信度兜底路径返回开发计划

关联设计：`docs/add_fallback_path_design.md`

## Stage 1：检索返回策略实现

### Task 1.1：调整 add 低置信度返回路径逻辑
- 文件：`tree_insertion_rag/retriever.py`
- 内容：
  - 将 `should_return_path` 改为按 action 分支控制。
  - 在 `add + low` 情况下返回 top1 的 `jsonpath` 与 `matched_node_id`。

### Task 1.2：补充 add 低置信度兜底 reason
- 文件：`tree_insertion_rag/retriever.py`
- 内容：
  - 明确低置信度兜底说明，便于调用方识别该路径为 fallback。

## Stage 2：测试与回归

### Task 2.1：新增/更新单测
- 文件：`tests/test_tree_insertion_rag.py`
- 内容：
  - 将 `add` 低置信度场景改为断言“非空路径 + low confidence”。
  - 保留其余行为断言不变。

### Task 2.2：执行测试
- 命令：`python -m unittest tests/test_tree_insertion_rag.py`
- 目标：通过所有相关用例。

## 提交计划
- 完成 Stage 1+2 后进行一次 git 提交，commit message 体现“add 低置信度兜底返回”。
