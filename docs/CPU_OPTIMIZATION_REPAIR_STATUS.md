# CPU 优化修复状态

> 执行计划：`docs/CPU_OPTIMIZATION_REPAIR_AGENT_PLAN.md`。状态仅在有可复查证据后更新。

| 阶段 | 状态 | 修改文件 | 验证/证据 | 备注 |
|---|---|---|---|---|
| R0 基线冻结 | verified | `local/cpu-repair/r0-baseline.json`, `local/cpu-repair/r0-working-tree-backup.zip` | HEAD `93ba16cb6c8d46564280b3459cdec731dcd2ef9f`; tracked diff SHA256 `4c562b5afb17f619b2389edb7c7dde28cea59d3b01633870248aa7662e4e5c5a`; backup SHA256 `353a19199e5429abfd1f3dd50964898ec55a0a804c06d21483b96e766c59467c` | 已保留初始 25 个 dirty 文件；用户 Vite PID 45732 保留。 |
| R1 / F1 内容身份去重 | verified | tools/audit_independent_holdout.py, tests/test_audit_independent_holdout.py | red/green；pytest → 28 passed | 所有相连重复成员排除；支持数只计唯一图像与组。 |
| R2 / F2 资产验证与实际加载绑定 | verified | tools/compare_cpu_backends.py, tests/test_backend_artifacts.py | 12 targeted tests passed；Ruff 通过；真实 OpenVINO 资产单次 reload 成功 | 实际消费 XML/BIN 路径及哈希均记录并匹配清单。 |
| R3 / F3 超时与完成状态 | verified | tools/profile_pipeline.py, tests/test_profile_pipeline.py | 定向 repair suite 55 passed；完整 suite 292 passed；AB/BA/AB profiler 12/12 tasks complete | Deadline cancel/join、冻结分母、保守处理失败和逐行 partial JSONL 均验证。 |
| R4 / F4 实际路径 profiler 插桩 | verified | tools/profile_pipeline.py, tests/test_profile_pipeline.py | legacy 与 decoded 两条源码路径，各完成固定 100 样本×3 单轮/模式；必需阶段无 missing；实际模块路径和 SHA 已记录 | 覆盖实际 DetectionPredictor NMS、旧/新 decode/runtime、特征、规则和 overlay；进程内 ASGI，不代表网络浏览器延迟。 |
| R5 / F5 轮换日程与严格聚合 | verified | tools/compare_cpu_backends.py, tests/test_compare_cpu_backends.py, tests/test_backend_artifacts.py | schema-v2 run 完整：18 正式 + 6 并发任务；每配置 pipeline 600 合计（false/true 各 300），顺序 API 600 合计（各 300 成功）；API 总计 920 行/配置 | `measurement_complete=true`；round stability pass；exact feature parity/performance gates 未通过，候选保持 `not_adopted`。 |
| R6 本地验证与正式重测 | verified | docs/CPU_PERFORMANCE_REPORT.md, docs/CPU_BACKEND_EXPERIMENT.md, docs/INDEPENDENT_EVALUATION_PROTOCOL.md, docs/CPU_OPTIMIZATION_STATUS.md, requirements-cpu-backend.txt, .github/workflows/ci.yml, tools/config_templates/compare_cpu_backends.example.json, results/cpu-repair/gate_summary.json | 全量 suite 292 passed / 84.55%；wheel+sdist、pip check、doctor、smoke-rules、Ruff、compileall 通过；独立后端安装验证 NumPy 2.3.5 | CI 已配置 Windows/Linux × Python 3.11/3.12 并触发所有分支；本次收尾提交的 hosted CI 需 push 后触发并通过。 |

## 基线限制

- 不清理、重置或覆盖现存工作区改动；原 O4 权威结果保持只读。
- 源数据来源、人工标注/规则复核与新独立 holdout 仍是外部证据门槛，工程修复不会替代这些证据。
- 真实推理、导出与正式性能测量串行执行；模型训练、量化和默认语义变更不在本计划范围。

## 最终门槛

- `code_fixes_verified=true`：F1–F5 回归和工具检查通过。
- `measurement_revalidated=true`：Profiler AB/BA/AB 和 schema-v2 后端矩阵完整，且报告与原始摘要一致。
- `maintenance_release_readiness=pending_ci`：完整 Windows/Linux × Python 3.11/3.12 workflow 已配置，当前提交的 hosted CI 尚未运行。
- `model_promotion_readiness=blocked`：源数据 provenance、独立 holdout 与人工复核继续阻塞模型晋级；它们不阻止默认模型和风险语义不变的维护版本在工程发布检查通过后单独交付。可选后端仍 `not_adopted`。证据与哈希汇总在 ignored `results/cpu-repair/gate_summary.json`。
