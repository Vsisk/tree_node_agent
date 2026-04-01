# Add 任务低置信度兜底路径返回设计

## 背景
当前检索流程在 `action="add"` 且最终置信度为 `low` 时，会返回：
- `jsonpath = None`
- `matched_node_id = None`

这会导致上游在新增节点场景没有可用落点，即使排序阶段已经有“相对最优”的候选节点。

## 需求详情
在 `add` 任务中，当没有“足够合适”的路径（即低置信度）时：
- 仍返回最有可能的路径（rank top1）
- 不再返回空路径

## 澄清结果
1. 兜底策略仅作用于 `add`。`modify/delete` 行为保持不变。
2. 触发兜底时，`confidence` 仍保留为 `low`。
3. `reason` 需要明确为低置信度兜底语义，避免调用方误判为高质量匹配。

## WHAT
- 调整 `TreeInsertionRetriever.find_best_node` 的返回策略：
  - 对 `add`：即便 `confidence="low"`，也返回 top1 的 `jsonpath/matched_node_id`。
  - 对非 `add`：维持现有 `low` 时返回空 path 的策略。
- 保持 `score`、`top_candidates`、`debug` 等输出结构兼容。
- 增补单元测试覆盖该回归场景。

## WHY
- 提升 `add` 场景可用性：保证始终有可执行候选路径。
- 同时保留风险信号：`confidence="low"` 不变，调用方可按需做二次确认。
- 兼顾兼容性：仅对 `add` 改动，降低对 `modify/delete` 的影响面。

## HOW
1. 在 `find_best_node` 中将“是否返回路径”的逻辑由单一 `confidence != low` 改为：
   - `action == add`：始终返回 top1 路径（前提是有 ranked candidate）。
   - 其他 action：保持原有规则。
2. `reason` 生成规则：
   - `add + low`：优先使用低置信度说明并追加兜底标记文案。
   - 其余：保持当前逻辑。
3. 在测试中新增 `add + low` 断言：
   - `jsonpath` 非空
   - `matched_node_id` 非空
   - `confidence == low`
4. 运行现有测试，确认无回归。
