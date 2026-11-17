# SafeSight 下一轮执行状态

最终同步分支：`codex/safesight-v030-final-status`（基于 origin/main `168bddcf3d22a28bbef15c8d7001404c03432c6e`）。PR #30 已合并；计划文件由用户提供，保持未跟踪且未修改。
已发布代码标签：`v0.2.3` 指向维护版本代码（PR #28 合并 SHA `b18c037e9d0318d66d0ea2ee82dd56c65f67fe55`）；正式 [v0.2.3 Release](https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/releases/tag/v0.2.3) 已公开，wheel、sdist、SHA256SUMS 匿名下载和校验通过。`v0.2.1` Release 已核实。
已发布代码版本：`v0.3.0`；模型资产仍独立使用 `v0.2.1`。正式 Release 与公开 wheel/sdist/SHA256SUMS 已核验。

## 环境、用户改动与进程

- Windows；Python 3.11.18（本机 Conda prefix `.venv/python.exe`）、Node 22.19.0、npm 11.6.0。
- 仓库无适用 `AGENTS.md`、`project-docs/index.html`、`DECISIONS.md`；本次重读本计划及本状态文件。
- 本轮 tracked 改动包括 v0.3.0 candidate 版本元数据、README/CHANGELOG/CITATION、复现进度和本状态文档；唯一未跟踪项仍为用户提供且未修改的 `docs/NEXT_ITERATION_AGENT_PLAN.md`。候选发布包与 smoke 日志均写入 Git 忽略的 results/logs。
- 仍有用户的 Vite 开发服务器运行（PID 45732，`127.0.0.1:5173`，父 `cmd.exe` PID 42228）；未停止。未发现 SafeSight Python/API 或 pytest 进程。
- `docs/NEXT_ITERATION_STATUS.md` 已纳入版本控制；仅本文件和上列测试文件由本轮改动。

## 阶段状态

| 阶段 | 状态 | 当前证据 / 未完成项 |
| --- | --- | --- |
| S0 | verified | 本机运行环境如上。M2 已合并到 `origin/main`；v0.2.3 公开 wheel 独立环境通过 pip check、doctor 和真实模型 smoke。 |
| S1 | verified | 提交 `5a97496`（完整 SHA 见 Git）；前次专项验证记录仍有效。 |
| S2 | verified | 提交 `9aaa3ec`；前次专项验证记录仍有效。 |
| S3 | verified | 维护代码 `v0.2.3` 已在 `origin/main` 并有 tag；PR #28 的 CI/Security/真实模型 smoke 在准确 head `1e3789a91f88efff7eb48e7f13fd6cefc638e22a` 通过，合并到 `b18c037`。正式 [v0.2.3 Release](https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/releases/tag/v0.2.3) 已公开并含 wheel、sdist、SHA256SUMS。匿名下载 hash 与本地构建一致：wheel `f399db1f1bdac09a74deac7a6e40877c8d509b30ddb1c6729f38c8bbb1808c09`，sdist `0b959806d98d477669a23951530e4c0db21540a7c00994be652ed0996ac6e48c`。公开 wheel 源码外安装后，pip check、verified YOLO 下载、doctor、readiness/API 与真实模型灰图 smoke 均通过。未发布私有模型或数据。 |
| S4 | verified | 提交 `ec9c2eb` 与审查修复 `168db78` 增加 provenance、跨 manifest ID/group/pixel audit、BDD 上下文/遮挡/截断、小/tiny 支持统计。实际有界准备输出 `results/next-iteration/s4-evaluation/`：1,000 张 val 选择，扫描 10,000 train，1,621 images、4,553 crops（训练上限内 1,924）；development/holdout 灯色分别 green/red/unknown/yellow=566/311/481/44 与 532/280/422/12，tiny TL=73/47；训练 yellow shortfall=76，evaluation yellow shortfall=56/88。与旧 manifest 的跨角色冲突 0、同角色重复记录 4,798。属性及来源虽可审计，但数据仍 `local-unverified`、annotation_review=`awaiting-human-review`，独立新 holdout 未建立，promotion_allowed=false。定向 15 tests、Ruff、diff check 通过。 |
| S5 | verified, exploratory | commits `1f5665e`、`dbafc58` 增加只读错误分析和固定参数 runner。新 S4 manifest 上选 200 development frames/200 ride groups（150 small/tiny-light、50 no-light），公开 YOLO11n、CPU/2 threads、8 配置均处理 200 张、0 failure、总计 173.48s。详细结果 `docs/S5_CPU_EXPERIMENT.md`；ignored JSONL/summary/bootstrap 与运行时源码快照在 `results/next-iteration/s5-cpu-experiment/`。200 paired ride-group bootstrap 给 traffic-light recall 区间，mAP 区间未计算。所有 960 / ROI detector-path P95 ratio >1.5；C2 的 mAP 无增益且 precision 下跌。计时不是完整 pipeline，1.5×全流程门槛未核验。来源仍 local-unverified、无独立新 holdout，因此 `promotion_allowed=false`，默认模型/配置保持不变。S5 focused tests 9 passed，Ruff、diff check通过。 |
| S6 | verified | 提交 `b1bcc73` 增加兼容的 opt-in detection API、reason_code 文案、归一化 SVG overlay 与可访问开关；无效坐标/分数不会序列化。后端聚焦 29 tests，Ruff/diff clean；前端 12 tests、lint/build 通过。浏览器本地端到端 API 200，`hard_rule_triggered`、11 detections、toggle 和换图清理通过；mobile portrait aspect ratio 通过。截图与脚本仅在 ignored `results/next-iteration/s6-browser-result.png`、`s6-browser-mobile.png`、`s6_browser_smoke.py`。 |
| S7 | verified | PR #30 合并到 `168bddcf3d22a28bbef15c8d7001404c03432c6e`；精确 head `be2c2c8443fcb2c78bbf60e95c4686f3e850afe8` 的 CI、安全、前端与 Windows/Linux 真实模型 smoke 全部通过。v0.3.0 tag/Release 已公开；匿名下载与 checksum 匹配。wheel 在源码外锁定 CPU 环境安装，pip check、doctor 与 smoke 通过。 |

## 本轮运行证据（2026-11-16）

日志：`logs/next-iteration/20261116-m2-validation.txt`（本地忽略）。

- `pytest tests/test_prepare_evaluation.py tests/test_analyze_detection_errors.py tests/test_app.py tests/test_predict_pipeline.py -o addopts='--strict-markers'`：阶段性重跑 exit 0，47 passed，1 Starlette/httpx 弃用警告（S4 15、S5 analysis 3、API/pipeline 29）。
- 当前新增 S5 focused tests: experiment 4 and paired bootstrap 1; analysis tests 3 (total 8 S5 tests), all exit 0. Frontend tests: 12 passed; `npm run lint`、`npm run build` exit 0。
- `ruff check`（本轮涉及 Python 文件）、`git diff --check`：exit 0。
- 首次全量 pytest 暴露 4 个旧 YOLO 测试替身缺少 `conf` 字段；补齐替身后 `.venv/python.exe -m pytest --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=80` exit 0，188 passed，84.32% coverage，3 项环境/弃用警告。
- 本地浏览器 e2e：API 200、11 个框、reason 显示、开关/换图清理、手机 portrait 比例通过；screen captures 在 ignored results。
- v0.2.3 公开 Release 的 wheel/sdist/SHA256SUMS 已匿名下载到 ignored `results/next-iteration/v0.2.3-public-verification/`；公开 hash 与本地 Release 构建文件匹配。该 wheel 源码外安装和真实 YOLO doctor/smoke 全部 exit 0。
- 无 SafeSight 后台进程。

## 本轮增量（2026-11-16，美国东部时间）

日志：`logs/next-iteration/20261116-s7-validation.txt`、`logs/next-iteration/20261116-s7-doctor-smoke.txt`（本地忽略）。

- PR #29 已于 2026-11-17 01:30 UTC 合并；origin/main SHA `3487aa8d3cfa5e8585f9c435742026e539bc80c6`，tree 与本地 PR head `557c1d789cbe15afa9b2ecdfc7b00f687ab5afb6` 完全相同。精确 head 的 9 项 workflow checks 全部成功：CI 四平台/版本矩阵、Security、Windows/Linux Public real-model smoke、Frontend、Gitleaks。
- 本机 Python 3.11.18：`pip check`、`compileall`、`ruff check src tests tools`、完整 pytest `188 passed` / `84.32%` coverage、`git diff --check` 均 exit 0。前端 `npm run lint`、`npm test -- --run`（12 passed）、`npm run build` 均 exit 0。
- `.venv/Scripts/traffic-risk.exe doctor --config local/config.json` exit 0；`smoke-rules --model-dir models` 使用真实 YOLO11n 权重完成 readiness + 灰图推理并返回预期 `no_supported_detections`，exit 0。该结果是真实模型灰图 smoke；PR 中的可控检测回归属于模拟输入，分开记录。
- `.gitignore` 忽略 `local/`、`results/`、`logs/`、`.venv/`、`.pytest_cache/`、`.coverage`、`risk_frontend/node_modules`；仅 `docs/NEXT_ITERATION_AGENT_PLAN.md` 留在 Git 未跟踪状态。
- 已公开 v0.2.3 Release 并上传 wheel、sdist、SHA256SUMS；匿名公开下载与 hash、安装、doctor、smoke 验证完成。
- Vite PID 45732 (`127.0.0.1:5173`) 仍运行；无 SafeSight API/pytest 进程。

## 最终状态

S0–S7 工程工作已完成。数据来源核实、人工 annotation review、独立新 holdout 与完整 pipeline latency gate 仍未完成；因此不晋级候选检测器，不改变规则优先默认配置，也不声称准确率提升。

## v0.3.0 候选构建记录（哈希已由最终发布记录更新）

- 当前分支 `codex/safesight-v0.3.0`，从 `origin/main`（`3487aa8d3cfa5e8585f9c435742026e539bc80c6`）起步。
- 准备 v0.3.0 candidate 元数据与发布说明；README 明确 source 为 local-unverified、没有准确性晋级声明，默认规则与 v0.2.1 模型资产不变；更新 CHANGELOG/CITATION/复现进度。
- `python -m build --wheel --sdist` 成功。忽略产物目录 `results/next-iteration/20261116-v030-dist/` 包含 wheel、sdist、SHA256SUMS。wheel SHA256 `fb84406ee61e4265ec847b050d317ac617b0e61563af2ac56337e6026ddbcf87`；sdist SHA256 `3fe06b029909ff2c58f032216ecf2cfb1b2c37edab000f67787e6a09230d1ca8`。
- `tools.wheel_smoke` 使用 CPU lock 在源码目录外创建隔离环境成功安装 wheel；pip check、配置默认路径/manifest schema、opencv 唯一发行包与 `traffic-risk --help` 检查通过（exit 0）。记录：`logs/next-iteration/20261116-v030-wheel-smoke.txt`。
- v0.3.0 metadata 改动后的本机完整 Python 回归 188 passed / 84.32%，pip check、Ruff、diff check 全通过；候选 wheel/sdist 在 `results/next-iteration/20261116-v030-dist/`，wheel smoke 证据 `logs/next-iteration/20261116-v030-wheel-smoke.txt`。v0.2.3 与 v0.3.0 正式 Release 均已公开并匿名核验。


## S7 最终发布记录（2026-11-17）

- PR #30 [release: prepare SafeSight v0.3.0](https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/pull/30) 已合并；合并 SHA `168bddcf3d22a28bbef15c8d7001404c03432c6e`，PR head SHA `be2c2c8443fcb2c78bbf60e95c4686f3e850afe8`。该 head 的 Security、Gitleaks、Frontend、Windows/Linux Python 3.11/3.12 CI 与真实模型 smoke 全部成功。
- 正式 [v0.3.0 Release](https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/releases/tag/v0.3.0) 已发布并标为 Latest，包含 wheel、sdist、`SHA256SUMS`。匿名下载保存于 ignored `results/next-iteration/20261116-v030-dist/public/`；wheel SHA256 `fb84406ee61e4265ec847b050d317ac617b0e61563af2ac56337e6026ddbcf87`，sdist SHA256 `3fe06b029909ff2c58f032216ecf2cfb1b2c37edab000f67787e6a09230d1ca8`，均与正式清单匹配。
- 从匿名下载的公开 wheel 使用 `requirements-cpu.lock` 安装在源码树外；`pip check`、CLI、显式隔离目录的 doctor 与公开 YOLO11n 下载均通过，真实模型 `smoke-rules` 得到 `no_supported_detections` 且 exit 0。完整记录：`logs/next-iteration/20261116-v030-public-wheel-validation.txt`。本机完整 Python 188 passed / 84.32%，前端 12 tests、lint/build，Ruff、compileall、pip check、diff check 均通过。
- v0.3.0 未改变 rules-first 默认、v0.2.1 YOLO11n 公开模型资产或风险语义。数据来源仍 local-unverified、人工复核待完成、无独立新 holdout，完整 pipeline 延迟门槛也未满足；promotion_allowed=false。
- v0.3.0 Release 页面是发布的最终证据；本地仍运行用户的 Vite 开发服务器（PID 45732，127.0.0.1:5173），未发现 SafeSight API/pytest 后台进程。
