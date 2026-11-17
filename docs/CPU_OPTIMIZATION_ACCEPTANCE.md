# CPU 优化修复独立验收记录

验收日期：2026-11-17（America/New_York）。

## 结论

**F1–F5 工程修复及当前 Windows/Python 3.11 环境下的测量复验通过。** 本次没有发现阻断这五项修复验收的新功能问题。可选后端继续不采用；维护版本发布就绪度由安装、完整 CI 和平台运行结果决定。来源证明、独立 holdout、人工标注/规则复核只阻塞模型晋级。

验收以实际工作区、测试及原始测量文件为准，没有仅依据执行 agent 的状态描述，也没有重新运行完整性能矩阵。验收时实现仍在 `codex/cpu-performance-evaluation` 工作区中，基线 HEAD 为 `93ba16cb6c8d46564280b3459cdec731dcd2ef9f`；该基线提交本身不包含本次优化实现。

## 独立检查结果

| 项目 | 本次核验 | 结论 |
| --- | --- | --- |
| F1 候选内容重复 | 重新构造一张真实内容、500 个样本 ID、100 个组 ID 的合成输入；返回 eligible=0、candidate_content_unique=false、holdout_eligible=false | 原失败案例已关闭 |
| F2 资产加载绑定 | 对上轮“清单中的 XML 与 artifact_root 分离”原复现文件重新运行校验；在加载前抛出父目录绑定错误；检查唯一 XML、同名 BIN、metadata 和加载边界代码 | 原失败案例已关闭 |
| F3 并发超时 | 重新构造顺序请求完成后首个并发请求超时；顶层 partial=true、原因 concurrency_timeout、completeness=false | 原失败案例已关闭 |
| F4 阶段插桩 | 检查新/旧入口与实际 DetectionPredictor.postprocess 的探针；逐一读取 AB/BA/AB 的 12 个任务，每项均有 100 条 pipeline 和 100 条 sequential API，无 missing 阶段、partial=false | 本次真实路径证据成立 |
| F5 轮换与聚合 | 检查冻结日程、18 个正式任务与 6 个后置压力任务；从逐请求 JSONL 独立检查样本/轮次唯一性、失败、分母，并重新计算各轮 P95 及其中位数 | 6 个配置全部与摘要一致 |

本次独立复跑：

- 完整 Python 回归：292 passed，3 warnings，覆盖率 84.55%。
- `ruff check src tests tools`、`compileall -q src tests tools`、`pip check`、`git diff --check` 均通过。
- 真实 YOLO `smoke-rules` 通过：readiness 正常，灰图得到预期 `no_supported_detections`，返回 `unknown`。
- `results/cpu-repair/gate_summary.json` 引用的 15 项文件哈希全部匹配，包括测试文件、测量摘要、构建包及默认 CPU 锁文件。构建包哈希核验不等同于本次重新完成源码外安装。

## 测量证据与正确分母

后端矩阵目录：`results/cpu-repair/backend-matrix-final-20261117/`。

- 冻结 schedule SHA256：`93f38cd59ae1383515d10424b97872f13b242fa49d4b72d41d3c7eab7920416c`，与 comparison 记录一致。
- 24 个任务状态全部 complete；当次工具 SHA 与当前 `tools/compare_cpu_backends.py` 一致。
- 每配置 pipeline 正式测量为 **600 条合计：false 300 + true 300**。
- 每配置 sequential API 为 **600 条合计：false 300 + true 300**，本次全部 HTTP 200 且无 processing failure。
- 每配置 API 总计 920 条：40 warmup + 600 sequential + 280 concurrency。
- 独立重算所有配置、两种模式、pipeline/API 的三轮 P95 中位数，全部与保存的 summary 一致。没有重新运行推理或挑选有利轮次。

Profiler 目录：`results/cpu-repair/profile-ab-ba-ab/`。12 项任务的 summary 哈希均匹配 run_status，固定样本测量完整。结果仍是单台工作站的本地观测，不是网络服务容量或跨平台性能保证。

## 发布前收尾状态

以下事项不推翻上述原始测量或 F1–F5 修复：

- **分母文案已更正。** 每配置两模式合计 pipeline 600 与顺序 API 600；false 和 true 各 300。原始逐行计数及 summary 无需改写。
- **独立后端依赖已显式固定 NumPy。** `requirements-cpu-backend.txt` 固定 `numpy==2.3.5`；默认 `requirements-cpu.lock` 仍保持 2.4.6。
- **跨平台 CI 已配置。** `.github/workflows/ci.yml` 覆盖 Windows/Linux × Python 3.11/3.12，并会在任意分支 push 时运行；当前提交的 hosted run 需由 push/PR 触发并通过后才能关闭此工程 gate。

## 尚未通过的门槛

- Windows/Linux × Python 3.11/3.12 workflow 已配置；本次修改提交后仍需取得 hosted CI 全绿结果，方可完成跨平台维护发布检查。
- 来源证明、人工标注/规则复核与新的独立 holdout 仍缺失，模型/检测配置晋级继续阻塞。
- **这些数据门槛不阻塞维护版本。** 默认模型与风险语义保持不变的维护版本可在工程发布检查通过后单独交付。
- 验收快照时未创建提交、PR、标签或 Release，也未改变默认模型、依赖锁或业务代码。

## 本次验收的工具源码身份

| 文件 | SHA256 |
| --- | --- |
| `tools/audit_independent_holdout.py` | `fe0ff51927c54eb834ffcf1e57f8e8d101f51755a91f7f18ca7dd46acbcb31a3` |
| `tools/compare_cpu_backends.py` | `9822510a8f5b367a7c0b707b27feda561f4f0032ba15425ef841d335f0ec74aa` |
| `tools/profile_pipeline.py` | `1ae7618543b2d490d9d5eae825a40a1212610afc7e093680fd102703d50e1bfe` |

后续修改工具、日程、依赖或模型资产时，应重跑受影响的检查；不要把本次验收直接套用到新的源码身份上。
