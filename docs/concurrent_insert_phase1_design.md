# 并发安全第一阶段插入机制设计（最小侵入）

## 1. 背景与问题

当前 `plan_nodes_by_json_path` 在单次调用中会基于传入的 `target_tree` 生成 `working_tree` 并完成路径补建与插入规划。
但在并发场景下，多个任务可能基于不同时间点的树快照进行规划，导致位置型路径（`children[index]`）语义漂移，出现“静默写错位置”。

用户已明确要求：
1. **不改对外接口**（仍保留 `plan_nodes_by_json_path(node, origin_tree, target_tree)`）。
2. 内部实现闭环版本管理。
3. 路径生成和节点插入分阶段。
4. 插入前在锁内更新到最新树并校验冲突。
5. 使用 `logging` 输出关键并发日志。
6. 路径生成阶段记录路径链路上的 `name+annotation`，提交前按该信息校验；不一致则重定位。

---

## 2. 澄清结论（需求确认）

### WHAT
- 在不改变对外接口的前提下，实现“分析阶段 + 提交阶段”的并发安全机制。
- 全程在提交关键区使用全树锁串行提交。
- 在提交前如果检测到树变化或路径链不一致，按最新树重算路径。
- 若重算失败，返回显式冲突/失败状态，不静默错误插入。

### WHY
- 位置型路径在并发修改后会漂移，旧 index 不可信。
- 通过锁+版本+链路校验可以显著降低错插风险。

### HOW（总览）
- 引入内部 `TreeStateManager`（模块内单例）：维护 `latest_working_tree`、`tree_version`、`threading.RLock`。
- 新增内部数据结构：`InsertPlan`、`CommitResult`、`PlanStatus`、`CommitStatus`。
- 将原流程拆为：
  1. `build_insert_plan`（只读、记录链路签名）
  2. `commit_insert_plan`（锁内，版本+链路复核，必要时重定位，再插入）
- `plan_nodes_by_json_path` 仍为对外主入口，但内部改为调用两阶段流程。

---

## 3. 详细设计

### 3.1 模块与职责（最小侵入）

仍在 `node_placement.py` 内完成，避免大规模重构：

1. **TreeStateManager（新增）**
   - 维护内部最新树与版本号。
   - `get_snapshot(target_tree) -> (snapshot_tree, version)`
   - `commit_with_lock(plan) -> CommitResult`

2. **Insert Planner（新增函数）**
   - `build_insert_plan(snapshot_tree, base_version, json_path, new_node, origin_tree)`
   - 只读：解析路径、定位父节点、计算候选插入位置。
   - 记录 `validation_summary`：沿路径每层 `name/annotation/node_type`。

3. **Insert Committer（新增函数）**
   - `commit_insert_plan(plan, origin_tree)`
   - 锁内读取最新树，检查版本和链路。
   - 若不一致则重定位：重新解析路径 + 重建父节点。
   - 成功插入后版本递增。

4. **现有能力复用**
   - `PathParser`, `ExistingTreeLocator`, `TreeBuilder`, `InsertPositionResolver`, `Deduplicator` 复用。

### 3.2 数据结构

#### InsertPlan
- `base_version: int`
- `target_json_path: str`
- `resolved_parent_path: str`
- `insert_index: int`
- `new_node: dict`
- `validation_summary: list[dict]`（每层 name/annotation/node_type）
- `status: str`（`ok` / `invalid_path` / `invalid_parent_type` / `plan_failed`）
- `message: str`

#### CommitResult
- `status: str`（`success` / `repositioned_and_success` / `conflict` / `invalid_path` / `invalid_parent_type` / `insert_failed`）
- `message: str`
- `applied_parent_path: str | None`
- `applied_index: int | None`
- `is_repositioned: bool`
- `working_tree: dict`
- `tree_version: int`

### 3.3 路径校验与重定位策略

1. 分析阶段记录路径链路签名（name+annotation+node_type）。
2. 提交阶段锁内：
   - 若 `current_version != base_version`，触发重定位。
   - 即使版本相同，也做轻量链路校验（防止外部绕过状态管理改树）。
3. 链路校验失败 -> 基于最新树重新执行路径解析与父节点解析。
4. 重定位失败 -> `conflict`。

### 3.4 日志策略（logging）

新增模块级 logger：
- 分析阶段日志：`base_version`, `target_json_path`, `resolved_parent_path`, `insert_index`, `plan_status`
- 提交阶段日志：`base_version`, `current_version`, `need_reposition`, `reposition_result`, `commit_status`

### 3.5 对外返回兼容

不改函数签名；返回结构在现有基础上扩展状态字段，确保上层能感知冲突：
- 已存在节点：`is_exist=True`
- 需插入：`is_exist=False`
- 额外附带 `commit_status`, `message`, `tree_version`。

---

## 4. 边界与限制（第一阶段）

1. 采用全树锁，吞吐不是最优，但正确性优先。
2. 不引入 node_id，仅使用路径链路签名与 name+annotation 去重。
3. 不做分布式锁，不做细粒度父节点锁。

---

## 5. 验收标准映射

- [x] 全局版本号递增
- [x] 全树写锁串行提交
- [x] 分析/提交两阶段拆分
- [x] 提交前版本校验
- [x] 版本变化后重定位
- [x] 冲突返回显式状态
- [x] 日志可观测（logging）
