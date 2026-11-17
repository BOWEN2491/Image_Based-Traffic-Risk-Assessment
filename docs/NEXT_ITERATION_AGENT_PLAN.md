# SafeSight 下一轮维护与优化：Agent 执行计划

创建日期：2026-11-16。状态：待执行。

基线：`main`，提交 `3a8cfae60f329b13f95a1ce5fe1d00897d6f7b64`，代码版本 `0.2.2`，PR #27 已合并。

本文是后续执行规范。本次编写计划不等于已经实施其中的改动。执行者开始时必须重新检查仓库、远端版本与适用指令；如果后续已有等价修复，以实际代码为准并记录差异，不要重复实现。

## 1. 目标、范围与完成定义

按顺序交付：

1. 修复真实模型 smoke 的错误通过风险，以及独立 CNN 推理入口与主运行时不一致的问题。
2. 发布一个可下载、可验证、可回退的维护版本，固化自动真实模型回归。
3. 补强数据来源与稀有场景评估，定位交通灯漏检原因，执行有边界的 CPU 优化实验。
4. 改进页面对 `unknown` 的解释，并提供与本次推理一致的检测框展示。
5. 全部变更通过本地及 Windows/Linux CI；留下可由另一 agent 接续的代码、证据和报告。

完成不要求一定更换模型。候选未达到质量、延迟、数据来源或样本支持门槛时，交付完整实验报告并明确保留基线，即为实验阶段完成。缺少独立标注时，不得宣称风险准确率提升。

### 1.1 范围约束

- 支持 Windows/Linux CPU，Python 3.11/3.12；前端沿用仓库声明的 Node 版本范围。
- 默认公开运行模式继续为 `rules`，只返回 `high/unknown`。本轮不恢复公开三级风险分类，不重新设计风险标签。
- 保持现有包名 `traffic_risk`、CLI 名称 `traffic-risk`、API 路由和六资产完整模型包布局。
- 本轮不全量训练检测器、不购买云资源、不使用外部 GPU，不迁移技术栈。
- 原始数据、历史桌面项目、历史实验结果只读。新实验必须使用新输出目录。
- 所有本机新增环境、缓存、临时文件、模型副本和输出放在当前 D 盘 workspace；公用代码不硬编码工作站路径。
- 本地数据、权重、截图、逐图预测、绝对路径和凭据不进入 Git 或 Release。
- 人工来源核实与标注不得由 agent 编造。无法取得时完成工具和已有证据部分，明确标记受阻的科学结论，继续不依赖它的工程任务。

### 1.2 里程碑

| 里程碑 | 内容 | 依赖 | 默认版本 |
| --- | --- | --- | --- |
| M1 | 原因码、严格 smoke、统一 CNN 入口、自动回归、正式维护发布 | S0–S3 | `v0.2.3` |
| M2 | 数据补强工具、漏检分析、CPU 参数比较、可解释页面 | S4–S6 | 待验收版本 `0.3.0` |
| M3 | 全套验证、最终文档、PR/Release 与工作副本同步 | S7 | M2 全部工程门槛通过后发布 |

版本默认值用于避免把新修复混入已验证的 0.2.2。若执行时该版本已存在，不重打标签；使用该版本序列下下一个未使用版本，并将依据记入执行状态。

## 2. 已知事实与证据

| 已确认事实 | 证据 / 影响 |
| --- | --- |
| 0.2.2 本地 137 项测试通过，总覆盖率 83.96%；运行时专项 85.64% | [复现进度](REPRODUCTION_PROGRESS.md)；这属于历史证据，新修改仍需验证 |
| PR #27 的 Windows/Linux × Python 3.11/3.12、前端和 Security 通过 | [PR #27](https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/pull/27)；执行时核验实际目标 SHA |
| smoke 只校验返回风险属于 high/unknown | [smoke_rules.py](../src/traffic_risk/smoke_rules.py)，异常也可能被 pipeline 转成 unknown |
| 独立灯色脚本固定 ResNet18，直接返回 argmax | [predict_tl_color.py](../src/traffic_risk/predict_tl_color.py)；主运行时已支持 MobileNetV3-Small 与 0.5 阈值 |
| 实际 pipeline 有正常 abstention 和错误两种 unknown | [pipeline.py](../src/traffic_risk/pipeline.py)；当前主要依赖人类可读 reason 区分 |
| 公开真实模型工作流仅手动触发、默认灰图 | [.github/workflows/model-smoke.yml](../.github/workflows/model-smoke.yml) |
| 正式公开 Release 检查时仍为 v0.2.1 | 发布前重新查询；不能仅凭 pyproject 版本认为已发布 |
| YOLO11n / YOLO11s 灯检测召回为 6.34% / 15.57% | [实验汇总](EXPERIMENT_RESULTS.md)；限定于这批本地留出集及既定阈值 |
| YOLO11s 反序复测 P95 为基线 1.707 倍 | 未通过原 1.5 倍延迟门槛；第一次计时有背景安装干扰，不得复用为升级证据 |
| MobileNet 灯色 macro-F1 低 0.28 个百分点，绿灯召回低 3.21 个百分点 | [CNN 结果](CNN_EXPERIMENT_RESULTS.md)；不支持直接替换 |
| 黄灯 holdout 只有 12 个；本地原始标注来源未独立认证 | [数据说明](EVALUATION_DATA.md)；科学结论与发布候选门槛均受限 |
| 旧人工 residual 标注已参与历史训练 | 不得用作新的独立风险测试集 |

## 3. Agent 执行方式与交接协议

### 3.1 角色及并行边界

- **协调 agent**：维护状态文件、决定阶段是否验收、协调共享文件、控制实验资源、准备 PR 和发布记录。
- **实现 agent**：每次领取一个阶段；实施功能并补回归测试，提交准确的变更与运行证据。
- **验证 agent**：独立检查实现、测试失败路径、核对数据和指标，给出通过或具体阻塞。
- 可以并行做只读调研、文档和互不重叠的文件；CPU 训练、性能测量、全套测试和大依赖安装必须串行。
- `cli.py`、`app.py`、版本文件、锁文件和状态文件各指定一个写入者；其余 agent 通过协调者交接修改。
- agent 数量受工具可用性约束；没有子代理时顺序执行同一协议，不以缺少并行能力为由省略验收。

### 3.2 持久状态与证据

S0 创建 `docs/NEXT_ITERATION_STATUS.md`，每阶段至少包含：状态、执行 SHA、改动文件、测试命令及退出码、证据路径、已知限制、下一步。

允许状态：`pending / running / verified / blocked / skipped-with-reason`。`verified` 必须有验收证据。实验结果差、候选未晋级并不等于阶段 blocked。

每个交接报告格式：

```text
阶段：Sx / 任务 ID
基线与当前 SHA：
已读来源：实际文件、接口、官方文档
实现内容与变更文件：
执行命令、退出码与关键结果：
证据：日志 / 测试 / 报告 / CI URL
未验证项或限制：
下一位 agent 的第一步：
仍运行的进程：PID、命令、输出路径（若无则写无）
```

本地日志放 `logs/next-iteration/`，实验放 `results/next-iteration/<run-id>/`，配置放 `local/`。状态文档只链接相对路径和可公开的聚合结论，不复制私有记录。中断恢复时先检查进程、输出和最后有效检查点；不得反复重跑已成功的昂贵步骤。

### 3.3 提交及操作边界

- 开始实现时创建独立分支；不 reset 或覆盖用户改动。
- 每个可独立验收的阶段形成清晰提交；不要为提升测试数字写实现镜像测试。
- 创建本计划文件本身不触发实现、推送、合并或发布。
- 后续执行遵循用户在会话中已经给出的授权及仓库保护规则；不要仅因本文包含发布步骤就推断不存在的权限。
- 对已授权的维护任务，无需反复询问常规实现选择。费用、数据访问许可、人工标注只能由真实提供者确认。
- 先完成可审查成果和所有可本地执行的验证，再处理任何确实需要外部权限的最后一步。

## 4. S0 — 重新核验基线与文档（必须先做）

**输入资料**：README、MODEL_CARD、docs/MODELS、docs/DEPENDENCY_UPDATES、docs/EXPERIMENT_RESULTS、三个现有 workflow；若存在 AGENTS.md 或 project-docs，先读它们。

**实施清单**：

- [ ] 记录 `git status`、HEAD、origin/main、Python/Node/依赖锁版本。读取 Release/PR/CI 状态，辨别代码版本、模型资产版本、正式 Release 三者。
- [ ] 检查本机 `.venv`：本仓库当前是 Conda prefix，解释器为 `.venv/python.exe`；普通 Windows venv 通常是 `.venv/Scripts/python.exe`，不要写死一种路径。
- [ ] 复用本机 `local/activate.ps1`；它不是新用户安装前提。可公开的初始化说明必须从 README 的干净环境开始。
- [ ] 核验 `.gitignore` 包含 local/results/logs/cache/env/coverage，执行 `git check-ignore` 证明。
- [ ] 创建状态文件、阶段分支，记录已有成功日志，避免把旧环境实验数字当作当前锁版本下的新测量。
- [ ] 列出实际可用 API；如工具或库签名改变，查本地已安装版本或官方文档后再实现。

**现有可复用接口**：

| 位置 | 接口 / 模式 | 后续使用 |
| --- | --- | --- |
| `pipeline.py` | `run_single_image(image_path, runtime) -> dict` | 统一业务结果，API 在外层添加 image_id |
| `app.py` | `create_app(settings=None, runtime_factory=ModelRuntime)`；`PredictionResponse` | 测试注入 runtime，新增字段必须同步响应模型 |
| `smoke_rules.py` | `smoke(model_dir, image=None)` | 复用真实 YOLO、ready 与 multipart API 路径 |
| `cnn_contract.py` | `contract`、`validate_contract`、`make_model`、`inference_transform` | 两架构共享加载与变换，不再手写另一套 |
| `cli.py` | 子命令分派、返回退出码 | 新命令必须兼容 `traffic-risk` 与 `python -m` |
| `tests/test_public_rules_detection.py` | 实际 detect→pipeline + 可控 YOLO 结果 | 灯＋近行人、仅灯、无检测的行为回归 |
| `tests/test_local_bundle_roundtrip.py` | 原生两类 CNN checkpoint + XGBoost UBJ + manifest 回加载 | 独立入口兼容测试的构造方式 |
| `prepare_evaluation.py` | `prepare(root, output, seed=42, ...)`；扫描及时间上限 | 固定样本身份、分组、裁剪与不足报告 |
| `evaluate_detection.py` / `tools/benchmark_latency.py` | 指标与多轮延迟报告 | 保留其指标定义，扩展参数比较 |

**验收**：本机 doctor、pip check 成功；状态文件包含真实基线、命令和未解决差异。发现旧问题已被修复时写证明并跳过对应实现，不删除验收要求。

## 5. S1 — 统一结果原因与严格真实模型 smoke（P1）

**目标**：正常 `unknown` 能通过 smoke，真实推理故障必须导致 smoke 非零退出，且评估工具与页面不再靠英文文案识别错误。

### 5.1 兼容接口

保持现有 `status/risk/reason/features/model_versions` 和 API 的 `image_id`。追加 `reason_code`；现有字段含义和 HTTP 状态暂不改变。

在轻量模块（建议新建 `inference_outcome.py`）集中定义原因码与 `is_processing_failure`，模块不导入 torch 或模型：

| reason_code | status/risk | 是否处理失败 |
| --- | --- | --- |
| `hard_rule_triggered` | ok/high | 否 |
| `model_prediction` | ok/low、medium、high | 否 |
| `no_supported_detections` | uncertain/unknown | 否 |
| `no_rule_triggered` | uncertain/unknown | 否 |
| `feature_schema_error` | uncertain/unknown | 是 |
| `perception_error` | uncertain/unknown | 是 |
| `invalid_features` | uncertain/unknown | 是 |
| `invalid_model_output` | uncertain/unknown | 是 |
| `legacy_unspecified` | 兼容旧响应 | 无法证明成功；新严格 smoke 按失败处理 |

新服务器的每条成功返回或捕获异常分支都显式设置原因码。`PredictionResponse.reason_code` 默认 `legacy_unspecified`，以便兼容旧调用者和旧测试桩，但服务器新路径不可依赖此默认值。现有 4xx/5xx 错误体保持不变。旧前端忽略附加字段；新前端缺少字段时显示原 reason。

### 5.2 实现要求

- [ ] 修改 pipeline 全部分支及 Pydantic 响应模型；为异常日志保留堆栈，但响应不泄露本地路径。
- [ ] 同步 `tools/real_model_smoke.py` 当前对 response keys 的严格集合断言：验证新的完整核心契约或必需字段子集，避免新增原因码后完整模型 smoke 误失败。检索所有 `set(response)`、JSON snapshots 和 mock payload，逐一核对兼容性。
- [ ] 修改 smoke：检查 readiness、mode=rules、status/risk 一致、reason_code 已知且不是失败；不比较完整英文 reason。
- [ ] 灰图运行必须得到 `no_supported_detections`。指定 `--image` 时接受正常 `no_supported_detections/no_rule_triggered/hard_rule_triggered`，不强行要求所有道路图都是 high。
- [ ] `smoke()` 故障抛出明确异常；CLI 和模块入口通过 `SystemExit(main())` 或未吞掉的异常返回非零。
- [ ] 评估工具和延迟工具使用共享分类函数。保留旧历史报告为只读；解析缺少原因码的旧报告时仅允许明确兼容逻辑，并记录使用了旧格式。
- [ ] 为线上默认 mode 缺失的旧测试桩补明确配置，不通过修改模型语义迁就错误 mock。

### 5.3 测试与验收

- [ ] 构造 PerceptionError、FeatureSchemaError、无效特征、非法模型输出：API 仍保持既有错误/unknown 契约，但 smoke 必须非零。
- [ ] 无目标、未触发规则、近行人 high 均正常；rules 永远不会因缺少 CNN 返回 low/medium。
- [ ] 同时测试 Python 调用、`traffic-risk smoke-rules` 和 `python -m traffic_risk.smoke_rules` 的退出码。
- [ ] 使用真实 YOLO 跑灰图和一张本地已验证道路图。记录权重 hash、SHA、mode、reason_code；不上传私有图片。
- [ ] 修改英文 reason 后分类结果不变；未知 reason_code 不得被 smoke 当作成功。

**反模式**：不把所有 unknown 改成 HTTP 500；不通过接受所有 HTTP 200 来证明模型成功；不把 `unknown` 当正常低风险；不取消当前隐私和资源控制。

## 6. S2 — 统一独立灯色推理入口（P1，可与 S1 非重叠文件并行）

**来源**：`predict_tl_color.py`、`cnn_contract.py`、`model_runtime.py`、`train_tl_cnn.py`、`cnn_experiment.py`、`cnn_detected_eval.py`；测试参考 `test_package_tools.py` 和原生 roundtrip 测试。

### 6.1 明确的兼容策略

- 保留 `load_model(model_path, mapping_path, device='cpu')` 返回 `(model, names)` 的方式，新增 keyword-only `metadata_path=None`。
- 显式 `--metadata FILE` 优先；未提供时读取 checkpoint 同目录 `model_metadata.json`。存在但内容非法时失败，不能回退掩盖错误。
- 两处都没有元数据时，仅按历史 ResNet18、128×128、固定四类映射兼容加载；输出清楚的 legacy 提示。禁止根据文件名或 state_dict 猜测 MobileNet 架构。
- 映射必须严格等于 `CLASS_MAPPING`。不再接受历史单测中的随意两类映射；为此更新测试并在变更说明记录该校验收紧。
- 保留 `--model/--mapping/--image/--device`。新增 `--metadata`、`--json`；默认人类可读格式保留。

### 6.2 实施

- [ ] 独立入口调用共享 `validate_contract/make_model/inference_transform`，保留 `weights_only=True` 和 `strict=True`。
- [ ] 抽取共享概率转标签策略：top probability `<0.5` 返回 unknown，`>=0.5` 返回对应标签；非有限值、错误维度和错误映射直接拒绝。
- [ ] runtime、独立脚本及实验主指标共用该策略。raw argmax 仅保留为明确命名的实验对照项。
- [ ] `predict_image` 原 tuple 结构保留；数值是最高类别概率，返回 unknown 时也不是“unknown 的风险概率”。JSON 中命名为 `top_probability`，另给 `label`、`raw_label`、`architecture`、`abstained`。
- [ ] 用 context manager 解码图片并转 RGB；坏图、不兼容 checkpoint、缺失文件给非零退出和简明错误。

### 6.3 验收

- [ ] 两架构各有真实保存/加载测试；同一 checkpoint、同一图像预处理结果和标签与 runtime 一致。
- [ ] 测试 0.49、0.50、0.51 的边界以及 raw top class 本身是 unknown 的情况。
- [ ] 测试 legacy ResNet18 成功、MobileNet 缺元数据失败、非法元数据失败、错误 class mapping 失败、RGB/灰度/RGBA 转换。
- [ ] 不以降低 strictness 或补零权重解决架构不匹配。
- [ ] 不因共用策略重新改写已发布实验报告；若重新评估，写新运行记录。

## 7. S3 — 自动真实回归与维护版正式发布（P1，依赖 S1/S2）

### 7.1 回归工作流

- [ ] 将现有 model-smoke 工作流保留手动触发，并增加每周一次 UTC 定时触发、版本标签触发；对修改 runtime、pipeline、smoke、契约、锁文件或 workflow 的 PR 增加路径过滤触发。
- [ ] 真实模型 job 使用 Windows/Linux、Python 3.11 的最小矩阵；普通 CI 继续 Windows/Linux × Python 3.11/3.12。限制权限为 contents:read，设置超时与并发取消。
- [ ] 缓存仅减少下载；每次加载仍验证 manifest/权重 hash。下载失败必须报告为资产获取失败，不得跳过 job 后显示成功。
- [ ] 公共 CI 的必需覆盖为：真实模型灰图前向＋API/readiness，和可控检测结果的“灯＋人/仅灯/空检测”回归。二者名称必须区分，不能把合成检测测试宣称为真实道路检测。
- [ ] 为自动道路样例增加 `tests/fixtures/README.md` 来源清单规范：作者、来源、许可、允许的使用范围、hash、期望检测不变量。只有已证实可公开分发的样例才能加入 Git；来源不清时保留本地 `--image` 验证并在发布说明披露公共道路样例缺口，不阻塞其他维护修复。
- [ ] 有合规公开道路样例后，固定 hash 验证目标类别或规则路径；不要锁定浮点框坐标和过窄置信度数值，也不要只断言 high/unknown。

### 7.2 发布流程

- [ ] 修正文档里“CI pending”的过时描述，并合并重复的最终验证章节；链接实际通过的 SHA 和工作流。
- [ ] 将 package、API、前端、CITATION 和 CHANGELOG 版本保持一致。维护版默认 v0.2.3，明确仍复用 v0.2.1 模型资产，这是合法的独立版本关系。
- [ ] 构建 wheel/sdist，在干净 CPU 环境及源码目录外验证。不得让 PYTHONPATH 或 editable 安装掩盖缺包数据。
- [ ] 创建 PR，等待正确 head SHA 的 CI、Security 和真实 smoke；失败先诊断，不把普通“可合并”状态当作门槛通过。
- [ ] 在已授权的发布范围内合并、创建不可移动的版本标签和正式 GitHub Release。发布 wheel、sdist、对应 SHA256SUMS 及安装命令；无需重新发布未变更的模型权重。
- [ ] 从公开 Release 下载构建产物，重新校验 checksum 和版本，运行 doctor 与 smoke。记录 URL、提交 SHA、文件 hash、验证环境。
- [ ] 不上传 BDD 派生 CNN/XGBoost、训练数据或私有道路图片。Release 不能引用本地磁盘作为用户必须补齐的公共依赖。

**完成标准**：正式 Release 可匿名访问，文档命令使用固定版本可运行，跨平台检查通过，正常 unknown 与异常不再混淆。不要为等待后续模型实验推迟已经独立通过的维护发布。

## 8. S4 — 数据来源与稀有场景补强（P2）

### 8.1 数据身份与来源

- [ ] 保留 `results/evaluation-bounded` 和旧报告不变。新数据使用独立 run-id、manifest 版本和 hash。
- [ ] 复用 rideID 分组、文件及像素 hash 去重；同时比较新旧训练、开发、holdout 清单。相同 ride 或重复内容不得跨角色。
- [ ] 添加可验证 provenance 记录：数据提供者、原始包标识、来源 URL、获取/转换记录、校验值及核验人/日期。仅“文件名像 BDD”或格式匹配不能升级来源状态。
- [ ] 来源状态至少分为 `local-unverified / source-verified`；独立人工标注另记 `annotation_review`，不把文件来源验证等同于标签准确性。
- [ ] 自动核对 image/label ID、尺寸、bbox 范围、坏图、重复文件；统计保留和排除原因。保留排除条目，不能静默只挑简单图片。

### 8.2 小目标与稀有样本

- [ ] 按原始框面积比例分桶：small `<0.001`、medium `[0.001,0.01)`、large `>=0.01`；另列原图框最短边 `<6px` 的 tiny 子集。
- [ ] 属性存在时按 day/night、天气、occluded 分层；属性缺失报告 unknown，不从文件名或图像主观猜测为真值。
- [ ] 灯色新增评估目标：development 和新独立 holdout 各至少 100 个黄灯实例；未达到时报告实际数量，20 个只是最低支持门槛，不是充分统计保证。
- [ ] 抽样时以 ride/group 分配集合，不能拆开同图不同灯凑配额。平衡灯色集与自然分布检测集分开报告，禁止把平衡集分数称总体道路表现。
- [ ] 保留既有至多 2,000 训练裁剪、10,000 张训练扫描、每次准备 30 分钟限制。达到上限后持久化清单及 shortfall，不无限扫描凑数量。
- [ ] 旧人工 residual 只用于追溯，不进入独立风险测试。补充标签需要人工审阅时，生成本地 review queue 和标注说明，状态设 awaiting-human-review。

### 8.3 Holdout 管理

已有 498 张 holdout 已被观察和比较，只能作为历史回归参照。S5 在已有 development 上设计候选；新增最终 holdout 必须在选择候选前固定并封存，不能在测试后改阈值再称独立结果。

若无法形成新的、来源和分组可信的 holdout，仍可完成开发集探索与历史回归，但将 promotion_allowed 固定为 false，并将证据缺口列入最终报告。

**验收**：固定清单可重复生成；跨集合分组/内容无泄漏；每类数量、来源状态、tiny 样本和 shortfall 可审计。任何未核实来源不会被自动提升为 verified。

## 9. S5 — 交通灯漏检分析与有界 CPU 优化（P2，依赖 S4 工具）

**来源**：[检测说明](DETECTION_EVALUATION.md)、[历史实验](EXPERIMENT_RESULTS.md)、`evaluate_detection.py`、`tools/benchmark_latency.py`。实现 Ultralytics 参数前核对当前安装版本的官方 predict 文档，不假设存在尚未验证的参数。

### 9.1 先分析失败再比较参数

- [ ] 输出按尺寸/昼夜/遮挡分桶的 GT 支持数、召回、精确率和检测覆盖率。
- [ ] 对最多 100 个漏检、低 IoU、错误类别、误检样例生成本地标注图册，保留 sample_id、group、预测与 GT；图册不入 Git。
- [ ] 区分未检测、置信度过滤、定位误差、类别映射不支持、坏图和处理失败。不能把全部低召回归因于网络规模。
- [ ] 失败率使用 S1 原因码；检测异常保留 GT 作为 FN，并将处理失败与普通 abstention 分别统计。

### 9.2 固定第一轮候选

权重都用现有公开 YOLO11n，CPU、两个 Torch 线程、seed=42；不重新训练。开发集固定最多 200 图，含 small/tiny 灯且保留非灯场景：

| 配置 | 图像策略 | imgsz | operational confidence |
| --- | --- | --- | --- |
| B0 | 全图基线 | 640 | 0.25 |
| C1 | 全图 | 960 | 0.25 |
| C2 | 全图 | 640 | 0.15 |
| C3 | 全图 | 960 | 0.15 |
| C4 | 全图基础检测＋上方 60% 区域额外灯检测 | 640 / 960 | 0.25 |
| C5 | 同 C4 | 640 / 960 | 0.15 |

额外裁剪只贡献 traffic light；其框坐标必须映射回原图，和全图灯框按同类 IoU=0.45 NMS 合并，其他类别只用全图结果。裁剪前后坐标、边界、重复框必须单测。全流程耗时包括第二次检测和合并。

AP 仍使用 confidence=0.001 的预测和原有 AP 定义；阈值候选主要影响 operational recall/precision，不把阈值降低产生的召回增长声称为 AP 提升。保留不是官方 COCO 实现的说明。

每配置最多 15 分钟，第一轮总计最多 90 分钟；超时记录部分结果，禁止与完整结果直接排名。先完成轻量轮，只有一个获选候选进入最多 1,000 张最终评估。

### 9.3 选择和验收门槛

开发集上的筛选顺序固定：先排除处理失败和超过 1.5 倍 P95 的配置，再按 traffic-light recall、整体 mAP50–95、较低延迟依次排序；精确率或 person/car recall 较基线下降超过 1 个百分点的候选排除。

最终晋级要求全部满足：

- 来源已核实，新 holdout 独立，关键类别至少 20 个实例；类别不足则结论仅为探索性。
- mAP50–95 提高至少 1 个百分点；person/car/traffic light 任一召回下降不超过 1 个百分点，灯 precision 下降不超过 1 个百分点。
- 完整 pipeline P95 不超过基线 1.5 倍；处理失败率不得增长。
- 计时模型加载之后进行，至少 5 次 warmup、同一固定 50 张图片、3 轮交替模型顺序。期间无其他训练、测试、安装；报告每轮 P50/P95、RSS 和机器配置。
- 对按 ride 分组的重采样给出指标差异区间；若置信区间跨越零，标记提升不确定，不自动晋级。

第一轮不新增 ONNX/OpenVINO 等推理后端依赖。若只有延迟门槛失败，将后端优化写为后续独立实验，不扩大本轮。

### 9.4 输出与反模式

- [ ] JSON/CSV 明细留本地，Git 内发布聚合报告、配置模板、指标定义和复现命令。
- [ ] 每份报告记录代码 SHA、完整配置、模型和数据 hash、Python/Torch/OpenCV/Pillow 版本、起止时间、失败和超时。
- [ ] 未通过任一条件则保留当前默认策略；不得为了“做出升级”事后放宽阈值。
- [ ] 即使通过，也通过独立 PR 修改默认检测配置，重跑公开 smoke/API/前端，不在实验脚本内静默更新运行时默认值。
- [ ] 不根据本次感知分数声称风险模型准确率提高。

## 10. S6 — 页面原因解释与检测框（P3，可在 S1 稳定后并行）

### 10.1 原因解释

- [ ] `App.jsx` 使用 reason_code：未检测到有效目标、未命中保守规则、感知失败、特征处理失败分别显示。
- [ ] rules 模式展示“规则评估：可输出 high/unknown”；不把“统计模型未启用”统一表述为置信度不足。
- [ ] 无 reason_code 的旧服务仍显示原 reason；未知新原因码提供通用文案及原 reason，不报前端异常。
- [ ] 失败原因显示可操作建议（换清晰图片、重试、检查服务）；不把检测置信度展示为风险概率。

### 10.2 API 增量契约

新增 `POST /api/predict?include_detections=true` 可选参数，默认 false。普通请求继续使用既有核心字段。启用时增加：

```json
{
  "reason_code": "no_rule_triggered",
  "image_size": {"width": 1280, "height": 720},
  "detections": [
    {"id": 0, "category": "car", "bbox": [0.1, 0.2, 0.4, 0.7],
     "confidence": 0.82, "traffic_light_color": null}
  ]
}
```

此处只是附加字段示例，不能替代完整 PredictionResponse。bbox 为相对于推理坐标图像宽高的归一化 xyxy，限制在 [0,1]。confidence 是检测器分数；仅 traffic light 填颜色，其余为 null。rules 下灯色为 unknown。

- [ ] 同步 Pydantic 和前端类型/校验，限制最多 300 个框，禁止非有限坐标或分数进入 JSON。
- [ ] 为 `run_single_image` 增加 keyword-only `include_detections=False`；API 透传显式 query 值，CLI 与现有调用默认保持关闭。为检测详情定义嵌套响应模型，不能仅用未验证的任意 dict 绕过上述边界。
- [ ] 默认不返回 detections/image_size（response model 可设可选字段并 exclude_none），为旧客户端保留默认响应形态。
- [ ] 从同一次 detect 获取展示信息，不重复运行 YOLO；复用已有框、ID 和 image decode 结果。
- [ ] 前后端统一 EXIF 方向：明确在服务端规范化 orientation 后用于推理，前端预览采用相同方向；测试 EXIF 旋转 JPEG，避免框叠错位置。
- [ ] 不返回磁盘路径、原图 base64 或服务器临时文件 URL；浏览器继续展示其本地上传预览。
- [ ] features 和风险判断使用同一批检测；显示层不得重新过滤后伪装成模型实际输入。

### 10.3 页面与验收

- [ ] 用 SVG 叠加层在现有预览上画框，提供显示/隐藏开关、类别和检测分数；适配缩放与移动布局。
- [ ] 无框、异常返回、取消、超时、换图时清除旧框，及时释放 Object URL。
- [ ] 通过可访问名称、状态文本和颜色之外的标签表达信息。
- [ ] 测试新旧 API 响应、各 reason_code、框坐标边界、EXIF、resize、多次换图、503 和取消。
- [ ] 浏览器真实上传完成一次验证；截图本地保留，报告不公开私有道路图片。

**完成标准**：用户能辨认普通 unknown 与处理失败，检测框与当前上传图片一致，默认公开语义及旧客户端兼容。

## 11. S7 — 最终验证、发布与交接

### 11.1 运行命令

以下是仓库现有可用命令。先进入仓库根目录并激活当前环境；若使用本机环境可 `. .\local\activate.ps1`。不要假设其他机器有 local 文件。

```powershell
python -m pip check
python -m compileall -q src tests tools
ruff check src tests tools
pytest --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=80
git diff --check
```

运行时 85% 门槛使用 `.github/workflows/ci.yml` 中的实际命令，并将 S1/S2 新行为测试纳入该 job。不能降低现有门槛。测试先聚焦失败路径，源码稳定后跑完整门槛，只有新增改动或失败才重复。

```powershell
Push-Location risk_frontend
npm ci
npm run lint
npm test
npm run build
Pop-Location
traffic-risk doctor --config local/config.json
traffic-risk smoke-rules --model-dir models
```

道路图片命令使用本机实际路径变量，不把它写入公开文档。离线验证应在依赖和权重齐全后阻断网络，确认推理不隐式联网；离线第一次下载本来就不可行，不计作模型失败。

wheel 验证先构建到当前运行的独立目录，选择本次准确版本 wheel（不要用 glob 的第一项拿到旧 wheel），再调用现有 `tools.wheel_smoke --lock requirements-cpu.lock`。源码外 subprocess 清除 PYTHONPATH 和工作站 RISK_* 干扰，只保留明确必需的临时/缓存目录。

依赖未变时无需刷新所有锁；若依赖有变，使用完整 resolver 和 pip check 验证兼容，重新审计并做 wheel smoke，说明审计跳过的二进制构建范围。

### 11.2 最终验收矩阵

| 项目 | 必须证据 |
| --- | --- |
| smoke 正常/故障区分 | 正常 abstention exit 0；处理异常、未知码和不兼容响应非零 |
| CNN 一致性 | 两架构、legacy、阈值边界、真实 checkpoint 与 runtime 对齐 |
| 跨平台 | Windows/Linux × Python 3.11/3.12 CI 成功，记录准确 head SHA |
| 前端 | lint/test/build、旧响应兼容、原因解释和框渲染、真实浏览器检查 |
| 安装产物 | 当前 wheel、源码外环境、完整资源、真实模型 smoke、版本一致 |
| 数据 | 来源状态、固定清单、ride/hash 隔离、类支持数、排除与人工待办 |
| 实验 | 固定候选、独立选择/测试、指标定义、计时隔离、资源上限、明确晋级决定 |
| 发布 | 固定 tag、Release、公开资产与 checksum 可获取、安装命令可跑 |
| 隐私与工作区 | 无数据/权重/路径/密钥混入提交，旧资料未改，工作区状态明确 |

### 11.3 结束报告

协调 agent 最终回答必须列出：

1. 哪些阶段真正完成，哪些证据或人工工作仍缺失。
2. 合并 PR、发布 URL、代码 SHA、安装入口。
3. 本地与远端验证结果，明确区分真实模型、模拟检测和实验指标。
4. 默认模型/配置是否改变及门槛依据；没有升级同样给出确定结论。
5. 报告、状态和本地证据位置；是否仍有后台进程。

不得将计划中的 checkbox、单次 smoke 或旧测试结果当作本轮已经完成的证据。

## 12. 可直接交给执行 Agent 的任务提示

```text
请执行 docs/NEXT_ITERATION_AGENT_PLAN.md。

先读适用 AGENTS.md、计划和 docs/NEXT_ITERATION_STATUS.md（若存在），核验当前代码与记录基线的差异。从第一个尚未验收的阶段继续，保留已有用户改动和历史数据。

先完成 S0–S3 的维护修复和正式发布，再完成 S4–S6 数据、实验和界面工作，最后做 S7。可以并行只读调研与非冲突实现；CPU 实验、全套测试和大依赖安装串行。

按计划给每阶段留下测试退出码、SHA、报告和交接状态。未通过科学晋级门槛时保留默认模型并交付报告；不要扩大训练规模、编造标签或擅自改变评估门槛。

写入、推送、发布和外部资源操作遵循当前会话授权与仓库规则。任何实际阻塞都明确说明，并继续完成不依赖该阻塞的工作。不要把只完成计划或写出工具当成已经执行并验证了实验。
```
