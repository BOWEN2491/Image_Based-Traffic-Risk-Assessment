# SafeSight 下一轮执行状态

执行分支：`codex/safesight-m2`（与旧记录的 `codex/safesight-next-iteration` 不同；保留现分支）
当前 HEAD：`dbafc58`（S5 实验与分析工具）；`origin/main`：`b18c037`（M2 分支 ahead 6 commits）。用户提供的计划文件仍未跟踪且未修改。
已发布代码标签：`v0.2.3` 指向 `origin/main`；公开正式 Release 尚未创建。`v0.2.1` Release 已核实。
当前代码版本：`0.2.3`；模型资产仍独立使用 `v0.2.1`。

## 环境、用户改动与进程

- Windows；Python 3.11.18（本机 Conda prefix `.venv/python.exe`）、Node 22.19.0、npm 11.6.0。
- 仓库无适用 `AGENTS.md`、`project-docs/index.html`、`DECISIONS.md`；本次重读本计划及本状态文件。
- 未发现 SafeSight Python/API、Vite 或测试后台进程；进程清单中的 Node 属 Codex/MCP 工具。
- 保留原有未提交修改：`risk_frontend/src/App.css`、`App.jsx`、`src/traffic_risk/app.py`、`pipeline.py`、`tests/test_app.py`，以及新增的 S5 分析文档/工具/测试和用户提供计划；验证期间另有 `src/traffic_risk/model_runtime.py`、`tests/test_predict_pipeline.py` 未提交修改和 `docs/API.md` 新文件出现，保持未改。另因 S6 文案变更，同步更新了 `risk_frontend/src/App.test.jsx` 的旧文案断言与 reason_code 测试输入；未提交任何工作区内容。
- `docs/NEXT_ITERATION_STATUS.md` 已纳入版本控制；仅本文件和上列测试文件由本轮改动。

## 阶段状态

| 阶段 | 状态 | 当前证据 / 未完成项 |
| --- | --- | --- |
| S0 | verified | 本机运行环境如上。M2 分支基于 `origin/main` 并领先 6 个提交；原 doctor、pip check、compileall、真实 smoke 证据见前次运行记录。分支名已有漂移，未切换分支。 |
| S1 | verified | 提交 `5a97496`（完整 SHA 见 Git）；前次专项验证记录仍有效。 |
| S2 | verified | 提交 `9aaa3ec`；前次专项验证记录仍有效。 |
| S3 | running | 维护代码 `v0.2.3` 已在 `origin/main` 并有 tag；PR #28 在精确 PR head `1e3789a91f88efff7eb48e7f13fd6cefc638e22a` 的 CI/Security/真实模型 smoke 均通过，代码随后 rebase-merge 到 `b18c037`。本地 wheel/sdist/SHA256SUMS 与源码外 wheel smoke 已准备并通过。GitHub API 确认 `v0.2.3` 正式 Release 仍 404；资产尚未公开上传或匿名下载核验。GitHub 连接器没有 Release 创建工具，`gh` 不可用；继续尝试网页发布时，IAB 未登录且 Edge 当前未显示 GitHub 会话，未输入或索取凭据。 |
| S4 | verified | 提交 `ec9c2eb` 与审查修复 `168db78` 增加 provenance、跨 manifest ID/group/pixel audit、BDD 上下文/遮挡/截断、小/tiny 支持统计。实际有界准备输出 `results/next-iteration/s4-evaluation/`：1,000 张 val 选择，扫描 10,000 train，1,621 images、4,553 crops（训练上限内 1,924）；development/holdout 灯色分别 green/red/unknown/yellow=566/311/481/44 与 532/280/422/12，tiny TL=73/47；训练 yellow shortfall=76，evaluation yellow shortfall=56/88。与旧 manifest 的跨角色冲突 0、同角色重复记录 4,798。属性及来源虽可审计，但数据仍 `local-unverified`、annotation_review=`awaiting-human-review`，独立新 holdout 未建立，promotion_allowed=false。定向 15 tests、Ruff、diff check 通过。 |
| S5 | verified, exploratory | commits `1f5665e`、`dbafc58` 增加只读错误分析和固定参数 runner。新 S4 manifest 上选 200 development frames/200 ride groups（150 small/tiny-light、50 no-light），公开 YOLO11n、CPU/2 threads、8 配置均处理 200 张、0 failure、总计 173.48s。详细结果 `docs/S5_CPU_EXPERIMENT.md`；ignored JSONL/summary/bootstrap 与运行时源码快照在 `results/next-iteration/s5-cpu-experiment/`。200 paired ride-group bootstrap 给 traffic-light recall 区间，mAP 区间未计算。所有 960 / ROI detector-path P95 ratio >1.5；C2 的 mAP 无增益且 precision 下跌。计时不是完整 pipeline，1.5×全流程门槛未核验。来源仍 local-unverified、无独立新 holdout，因此 `promotion_allowed=false`，默认模型/配置保持不变。S5 focused tests 9 passed，Ruff、diff check通过。 |
| S6 | verified | 提交 `b1bcc73` 增加兼容的 opt-in detection API、reason_code 文案、归一化 SVG overlay 与可访问开关；无效坐标/分数不会序列化。后端聚焦 29 tests，Ruff/diff clean；前端 12 tests、lint/build 通过。浏览器本地端到端 API 200，`hard_rule_triggered`、11 detections、toggle 和换图清理通过；mobile portrait aspect ratio 通过。截图与脚本仅在 ignored `results/next-iteration/s6-browser-result.png`、`s6-browser-mobile.png`、`s6_browser_smoke.py`。 |
| S7 | running | 全量 Python 测试通过，M2 分支 push/PR、head-SHA CI/Security/真实模型 smoke、merge 与最终报告仍待办；v0.2.3 正式 Release 公开上传受 GitHub 登录会话阻塞。 |

## 本轮运行证据（2026-11-16）

日志：`logs/next-iteration/20261116-m2-validation.txt`（本地忽略）。

- `pytest tests/test_prepare_evaluation.py tests/test_analyze_detection_errors.py tests/test_app.py tests/test_predict_pipeline.py -o addopts='--strict-markers'`：阶段性重跑 exit 0，47 passed，1 Starlette/httpx 弃用警告（S4 15、S5 analysis 3、API/pipeline 29）。
- 当前新增 S5 focused tests: experiment 4 and paired bootstrap 1; analysis tests 3 (total 8 S5 tests), all exit 0. Frontend tests: 12 passed; `npm run lint`、`npm run build` exit 0。
- `ruff check`（本轮涉及 Python 文件）、`git diff --check`：exit 0。
- 首次全量 pytest 暴露 4 个旧 YOLO 测试替身缺少 `conf` 字段；补齐替身后 `.venv/python.exe -m pytest --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=80` exit 0，188 passed，84.32% coverage，3 项环境/弃用警告。
- 本地浏览器 e2e：API 200、11 个框、reason 显示、开关/换图清理、手机 portrait 比例通过；screen captures 在 ignored results。
- 公开 GitHub API：Release tag `v0.2.1` 返回正式 Release；tag `v0.2.3` 返回 404。网页/发布内容未创建或上传。
- 无 SafeSight 后台进程。

## 下一步

1. 继续 S3：建立 v0.2.3 正式 Release 并上传本地 wheel/sdist/SHA256SUMS，校验公开下载；等待与准确 SHA 匹配的 CI、Security、真实模型 smoke。网页目前需要 GitHub 登录。
2. S5 exploratory work is complete; do not promote without source verification, independent holdout, and complete-pipeline timing.
3. S7: local full Python suite passed; push M2 branch, create PR, wait for head-SHA CI/Security/model smoke, merge if green, then finish release and public asset checks. v0.2.3 Release remains the current blocker.
