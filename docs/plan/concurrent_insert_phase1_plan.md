# 并发安全第一阶段开发计划

## Stage 1: 基础并发状态管理

### Task 1.1
- 在 `node_placement.py` 新增 `TreeStateManager`（内部单例）
- 维护：`latest_working_tree` / `tree_version` / `RLock`

### Task 1.2
- 新增 `InsertPlan` / `CommitResult` 数据结构与状态枚举（或常量）
- 新增模块级 `logging` logger

### Task 1.3
- 增加基础单测：版本初始化、版本递增、锁内提交基本路径

---

## Stage 2: 两阶段流程改造（不改对外接口）

### Task 2.1
- 新增 `build_insert_plan`（只读）
- 复用现有解析/定位逻辑，记录 validation_summary

### Task 2.2
- 新增 `commit_insert_plan`（锁内）
- 实现版本校验、链路校验、必要时重定位、提交插入

### Task 2.3
- 将 `plan_nodes_by_json_path` 改为“调用两阶段”，保持签名不变

### Task 2.4
- 增加单测：
  - 单线程成功插入
  - 版本变化后可重定位并成功
  - 版本变化后重定位失败返回 conflict
  - 非法父节点类型
  - 非法路径

---

## Stage 3: 返回结构与日志收敛

### Task 3.1
- 在当前返回体中补充 `commit_status/message/tree_version`
- 保持 `is_exist` 分支兼容

### Task 3.2
- 完善关键日志：
  - 分析版本 vs 提交版本
  - 是否重定位
  - 重定位结果
  - 最终提交状态

### Task 3.3
- 回归测试 + 代码整理 + 文档更新

---

## 风险与决策点

1. **外部直接传入不同 target_tree**：
   - 采用“内部状态优先 + 首次对齐”策略，后续由内部版本驱动。
2. **返回结构兼容性**：
   - 不改签名，字段增量扩展。
3. **重定位策略失败条件**：
   - 路径解析失败 / 父节点类型不合法 / 链路签名不匹配。
