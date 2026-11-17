# SafeSight CPU 优化审查修复计划（Agent 可执行）

> 状态：待执行；本文件不是修复完成记录。
>
> 日期：2026-11-17（America/New_York）。工作目录：`D:\Codes\SafeSight`。
>
> 修复对象：`codex/cpu-performance-evaluation` 上尚未提交的 CPU 优化实现。检查时 HEAD 为 `93ba16cb6c8d46564280b3459cdec731dcd2ef9f`，实际被审查代码还包含工作区修改和新增文件，不能用该 SHA 单独代表实现。

## 1. 目标、范围与完成定义

本轮修复五个已经定位的问题，补齐失败测试、资产校验和测量流程，然后重新生成受影响的证据。代码修复不依赖人工补齐数据来源；质量晋级依然受独立数据和人工复核门槛约束。

| ID | 优先级 | 问题 | 验收结果 |
| --- | --- | --- | --- |
| F1 | P1 | 候选 holdout 内部同图不同 ID 未排除，可能虚增样本量 | 重复内容不进入支持数，不能通过 holdout eligibility |
| F2 | P1 | OpenVINO 校验的 model_file 与实际加载目录可分离 | 实际读取的模型及关联文件全部绑定并校验 |
| F3 | P2 | 并发测速超时仅标子项，顶层可能仍报告完成 | 任一必需测量未完成均 partial，并返回非零退出码 |
| F4 | P2 | Profiler 拦截旧接口和错误的 postprocess 层 | 新/旧路径实际阶段可观测，缺失阶段不会被当作零耗时 |
| F5 | P2 | 每配置连续三轮，未执行交替对照 | 按预先冻结的轮换日程串行测量，严格检查轮次和分母 |

本轮保持默认 PyTorch、YOLO11n、640、confidence=.25、`high/unknown` 语义。不得为了让实验通过而扩大容差、修改规则阈值、训练模型、量化、引入默认可选后端或降低数据门槛。

完成分为三个独立结论：

- `code_fixes_verified`：F1–F5 修复和回归已通过。
- `measurement_revalidated`：使用修复工具完成新测量、数据完整、报告与原始结果一致；不意味着候选一定更快或被采用。
- `release_readiness`：干净安装、所声明平台矩阵及发布所需检查通过。缺少 Linux/Python 3.12 证据时只能报告前两项的实际状态。

数据来源、人工标注/规则复核、新独立 holdout 缺失，仍应标记 blocked；它们不能作为拖延 F1–F5 工程修复的理由，也不能被工程测试代替。

## 2. 执行顺序与工作区纪律

按 **R0 → R1/R2 → R3 → R4 → R5 → R6** 执行。每项先加入可失败的最小回归，再实现修复，不先运行整套真实模型实验。

- 保留当前全部未提交工作、两份旧计划、历史报告和结果目录；禁止 reset、clean、覆盖旧 run 或清理桌面/数据目录。
- 继续当前合适的 `codex/` 分支；不要为了新计划丢弃现有实现。R0 记录完整 diff 和新增文件 SHA，备份至忽略目录。
- 开发、环境、缓存、实验与复现夹具均在 workspace 内；真实数据原目录只读。
- 创建 `docs/CPU_OPTIMIZATION_REPAIR_STATUS.md`，按 R0–R6 记录状态、修改文件、测试命令、退出码、证据及阻塞。允许状态：pending/running/verified/blocked/not_adopted。
- 如使用多个 agent：一名负责 R1，一名负责 R2，主 agent 负责 R3/R4；R2 完成后才能由一个写入者修改 R5 的同一比较工具。另一个 agent 可只读审查。真实推理/导出/性能测量始终串行。
- 资源保持 seed=42、CPU threads=2、batch=1。正式后端矩阵最多 90 分钟、每 backend/size 的全部子任务累计最多 15 分钟；普通 profiler 复验最多 30 分钟。超限保存 partial，不缩小分母后称完成。
- 本计划要求实现、验证和生成可审阅产物。提交、远端 PR、合并及 Release 按执行会话已有授权处理；单凭本文不发布版本、不管理任何定时自动化。

## 3. R0：文档、接口和失败证据核验

### 实施

阅读适用仓库指令、[原优化计划](CPU_PERFORMANCE_AND_EVALUATION_AGENT_PLAN.md)、[执行状态](CPU_OPTIMIZATION_STATUS.md)、[性能报告](CPU_PERFORMANCE_REPORT.md)、[后端报告](CPU_BACKEND_EXPERIMENT.md)、[独立评估协议](INDEPENDENT_EVALUATION_PROTOCOL.md)。核验当前代码，行号只作为本次审查定位参考。

| 真实入口 | 来源 | 本轮用法 |
| --- | --- | --- |
| `audit(candidate_manifest, history_manifests, provenance_path, dataset_root, output, *, max_rows_per_manifest=..., max_seconds=...)` | `tools/audit_independent_holdout.py` | 保留 CLI 与历史暴露审计，补内部去重 |
| `_validate_unique_candidates(rows)`、`_sample_support(rows)` | 同文件约 247、176 行 | 不能只按 sample_id 去重后宣称内容唯一 |
| `verify_artifact_manifest(path, *, backend, imgsz, source_sha256)`、`_load_model(...)` | `tools/compare_cpu_backends.py` 约 60、562 行 | 绑定验证结果与实际加载路径 |
| `_profile_api(...)`、`run(...)`、`main(...)` | `tools/profile_pipeline.py` 约 305、490、634 行 | 统一 partial、分母与退出码 |
| `_Probe.install()`、`restore()`、`_ACTIVE_RECORD` | 同文件约 221 行 | 精确拦截新/旧实际入口，保留请求隔离 |
| `compare(...)`、`_worker(...)`、`_stats(...)`、`_api_completeness(...)`、`adoption_completeness(...)` | `tools/compare_cpu_backends.py` | 改为父进程按全局轮次调度，并验证任务账本 |
| `DetectionPredictor.postprocess` | 锁定 Ultralytics 的 `models/yolo/detect/predict.py` | 真正的检测后处理入口；不是 BasePredictor 的占位实现 |
| `AutoBackend._model_type`、`OpenVINOBackend.load_model` | 锁定包的 `nn/autobackend.py`、`nn/backends/openvino.py` | 外层按名称识别后端，目录中用首个 XML 并读取同名 BIN/metadata |

必须重新核对实际后端隔离环境中的 Ultralytics 版本及源码。当前锁定版本 8.4.126 的普通 `.xml` 文件名不被外层 `_model_type` 识别；不能仅因底层 `core.read_model` 接受文件就改成 `YOLO('model.xml')`。

已有失败证据（只作线索，正式回归须用测试夹具重建）：

- 忽略目录 `results/cpu-optimization/review-synthetic-3mkbkji6/`：合成人工声明夹具下，1 张实际图片被写成 500 条独立 sample_id、100 个 group_id，审计输出 `holdout_eligible=true`。
- 同目录的假模型 bundle：哈希清单包含 `verified/model.xml`、`.bin`，`artifact_root` 却指向另一目录，验证仍返回该另一目录为加载入口。这是结构复现，没有执行假模型。
- 受控超时复现：顺序请求完成后，第一个并发请求超时；并发项 partial 为 true，顶层 partial 为 false。
- O2 `20261116-o2-torch640-with-detections-01/summary.json` 的 pipeline stages 只有 preprocess、forward、imread；检测总段、特征、overlay、postprocess 均缺失。
- `compare_cpu_backends.py` 当前按 size/backend 的固定列表启动 worker，每个 worker 内连续完成三轮。

### 验证与防错

- [ ] 记录 HEAD、dirty 文件及 SHA；测试夹具中的 Reviewer 等字段明确标记为合成，不写进真实 provenance。
- [ ] 找到实际 Python 和后端环境；不更新默认依赖锁来凑环境。
- [ ] 不重复安装或重跑已通过的大实验。后续所有接口新增项必须先实现，再作为命令使用。

## 4. R1 / F1：候选集合内部内容去重

### 实施

1. 从 `_validate_unique_candidates`、`_identity`、已有候选文件/像素重算逻辑出发，建立候选集合内部索引：sample_id 使用 source_namespace；image_sha256、pixel_sha256 全局比较。
2. 发现重复时，**将该重复关联组内所有候选排除**，而不是只排除后出现的一条。重复 ID 和内容重复可形成关联组，输出稳定的成员索引、触发字段和重复组标识。
3. 同组的不同图片不等于重复，不能对 group_id 做唯一约束；group_id 用于来源组统计。相同内容即使分配不同 namespace/group_id 也仍是内容重复。
4. eligible 行必须经过文件验证、完整身份、可靠分组、历史暴露排除和内部重复排除后，才能传入 `_sample_support`。支持数只来自剩余唯一内容，不计入已排除行。
5. audit 记录单列 `candidate_content_unique`、重复组/行数量和 blocker。只要输入存在未解决重复，正式 `holdout_eligible=false`；禁止用“人工已签核”覆盖机器重复检查。
6. 新字段以兼容扩展加入审计 JSON，保留现有字段和 `promotion_allowed=false`；更新独立评估协议说明内部重复处理方式。

### 必须新增的回归

- 不同 sample_id、相同文件 hash；不同编码文件、相同像素 hash；跨 namespace 的相同内容：均排除全部相关候选。
- 500 行指向一张图且伪装成 100 组：内容唯一性及 sample_size gate 失败，不能合格。
- 同一个可靠 group 中的两张不同图片：不能误删；无重复时支持数不变。
- 重复 sample_id、已有 same-role 历史 holdout、缺失字段、损坏图片、哈希错误原用例继续通过。
- 输入次序变化时，排除的身份集合和统计不变；相同输入重复运行输出确定。

**模式来源：** `tests/test_audit_independent_holdout.py` 的 `_row`、`_write_jsonl`、`_provenance`、证据齐全夹具和重复 ID 测试。只在 tmp_path 合成图片和声明，不访问真实数据。

**完成标准：** 修复前的新测试确实失败，修复后通过；支持数不会被重复内容放大；原历史审计行为保持。

## 5. R2 / F2：绑定资产校验和实际加载

### 实施

1. 保留锁定 Ultralytics 支持的 OpenVINO 目录入口，不修改第三方源码或默认模型 loader。
2. `model_file` 必须是已登记、已哈希的 `.xml`；解析后的 `artifact_root` 必须等于该 XML 的父目录。根路径始终位于派生 bundle 内。
3. 在实际加载目录枚举 XML，要求恰好一个且就是 `model_file`。同名 `.bin` 必须存在、在清单中并通过大小/hash；不能用目录里任意一个 BIN 满足检查。
4. 该 loader 会读取同目录 `metadata.yaml`，因此实际存在/使用的 metadata 必须被登记并校验。当前工具导出的标准 bundle 要求此文件；缺失或未登记时给出明确错误，不能静默使用默认类别名。核对 names、task、batch、imgsz 和 dynamic/NMS 参数与源模型及实验契约一致，不能只验证字节未变却接受互相冲突的声明。
5. 使用规范化后的路径比较、处理大小写和 `.`/`..`、拒绝越界及重复规范路径；符号链接/Windows junction 指向 bundle 外也必须拒绝。无需重写通用文件系统框架，复用现有安全解析方式。
6. 验证结果返回明确的 `model_path`、配套文件与 `load_path`。加载前立即复核这些绑定；记录实际交给 loader 的目录及唯一 XML 身份，拒绝目录中存在额外未登记模型。可复用比较工具现有 `LimitedCore` 包装器，在 `read_model` 边界断言/记录实际 XML、BIN，而不改第三方源码。
7. ONNX 保留显式文件加载；如检查发现 external tensor data 引用，要求引用全部安全地解析并哈希入册，或明确拒绝该形式。本轮导出策略不需要为了兼容未知产物扩大范围。
8. 检查当前导出的四份资产是否满足新契约：满足则复用且重新验证，不无条件重导出；不满足则说明具体缺项，不能只改 manifest 的 hash 掩盖来源变化。

### 必须新增的回归

- 清单校验 A，但 artifact_root 指向 B；同一目录多 XML；正确 XML 配错误同名 BIN；未登记/被修改 metadata：均在 YOLO 初始化前失败。
- XML/BIN/metadata 任一缺失、大小错/hash 错、规范路径重复、目录越界：明确拒绝。
- 有效最小 bundle：验证返回唯一绑定路径；通过 spy loader 断言加载参数只能指向该目录，不需要把假 XML 交给真实 OpenVINO。
- 在已锁定后端环境对真实有效 bundle 回加载一次，确认实际读取文件和记录一致；不重新运行完整性能矩阵。

**模式来源：** `verify_artifact_manifest`、`_load_model`、`tools/export_cpu_backend.py` 的 manifest 写入，以及当前 Ultralytics loader 源码。可新增 `tests/test_backend_artifacts.py`，与纯统计测试分开。

**完成标准：** 结构复现被拒绝、真实资产可加载、实际消费的关联文件全部受清单约束。不要宣称仅靠一次预检查可以防御任意并发恶意改写；本地实验资产按本轮不可变输入管理。

## 6. R3 / F3：超时和完成状态一致

### 实施

1. `_profile_api` 内任一 warmup、sequential、concurrency 阶段超时，都设置顶层 `partial=true`、阶段及原因，并保留已完成请求。
2. 对并发任务统一清理：超时后取消并等待本批未结束协程，记录取消/失败数量；不能只从外层循环 break 后让后台请求继续写结果。
3. 在 `run` 汇总时同时检查子阶段状态和预声明的完成任务/样本数，不只读一个顶层布尔值。任一必需阶段缺失或 partial 均使顶层不完整。
4. 明确退出码：0=必需测量完整且工具执行成功；3=预算或缺失测量导致 partial；2=配置/运行错误。候选未通过性能 gate 与工具没跑完是不同结果。
5. 未知/缺失 reason_code 复用 `is_processing_failure` 的保守分类，不另建只包含已知错误码的白名单；HTTP 200 不代表管线成功。
6. partial run 可以有分位数供诊断，但必须标明样本数及 partial；任何采用门槛都不得消费它。原始请求逐批或逐行落盘，父进程超时也应留住已完成记录。

### 必须新增的回归

- 顺序测量已完成、首个并发请求超时：子项和顶层均 partial，CLI 返回 3。
- 在 warmup、sequential、2/4 客户端中途分别超时；均保留完成行，不残留未等待的任务。
- 表面 `partial=false` 但少一张图/少一轮/少一个并发阶段：不能判完成。
- 全部完成，包括允许的并发 503：计数正确；503 不加入成功延迟分位数。
- HTTP 200 的 processing failure 或未知 reason_code 单列失败，不能作为健康结果通过。

**模式来源：** `tests/test_profile_pipeline.py`、`_profile_api` 的假 ASGI/客户端模式、已有 API busy/cleanup 测试。使用可控时钟和 fake client，禁止用真实长 sleep 触发超时。

## 7. R4 / F4：修复实际调用路径上的插桩

### 实施

1. 为新 decoded 路径和旧 path 路径建立明确的探针表。新路径测 `ModelRuntime.detect_image`、`pipeline.compute_features`、`_detection_details_from_size`；旧路径保留 `detect`、`build_features_from_bdd`、`_detection_details`。
2. 必须 patch 调用点实际引用的函数，而非仅 patch 定义模块的同名对象；核验 `from ... import ...` 的绑定。新旧适配器嵌套时记录父子关系，不把它们相加成额外检测时间。
3. 将后处理探针装在实际 `DetectionPredictor.postprocess`，标注为包含 NMS/结果整理的后处理段；不要把 BasePredictor 的未调用方法当作已测量 NMS。
4. 在本地 profiler 中补规则/信号策略段。保持公开响应不含计时字段。每请求用现有 ContextVar 隔离，任意异常退出后 restore 全部 monkeypatch。
5. 每个阶段输出调用数、适用条件和观测状态：`measured / not_applicable / missing`。无目标时特征/规则可以 not_applicable；开框且成功时 overlay 不应 missing。缺失不能用数值 0 表示。
6. 验证至少一张有支持目标的真实开发图片和一张空检测输入，两种 include_detections 模式都覆盖；灰图 smoke 不能独自证明特征/规则探针工作。
7. 收紧测量命名：当前 `httpx.ASGITransport` 是进程内 API 测量，不包含网络 socket/browser。沿用此名称及限制；不要把它升级为网络端到端。若时间在 `response.json()` 之前结束，文档不能称包含 JSON 解析。
8. 保存实际导入模块的路径身份与源码 SHA，包括新 `image_io.py`、特征/规则函数；不能用工作区文件 hash 冒充另一个 wheel 环境的实际执行代码。

### 必须新增的回归

- 可控时钟断言每个新路径阶段被调用，旧接口 stub 断言未被新路径使用；验证 postprocess 子类 override 被拦截。
- legacy runtime 仍有对应探针；无框模式明确 not_applicable，开框模式计数正确。
- 同一请求一次完整像素解码、一次预测；两个并发请求的阶段记录不混淆。
- 异常后原函数身份恢复；再运行一次不会叠加 wrapper。总时间由外层计时，不相加阶段 P95。
- 阶段覆盖缺失时 completeness/gate 不得通过；测试不断言机器相关的固定毫秒数。

**模式来源：** `_Probe`、`_stage`、`_wrap_sync/_wrap_async`、`tests/test_predict_pipeline.py` 的 decoded/legacy 用例和真实 `DetectionPredictor`。正式性能测量留到 R6；此阶段仅最小真实探针验证。

## 8. R5 / F5：按轮次调度，严格聚合

### 日程与统计口径（本计划固定选择）

不通过简单重排六个配置来解决问题。父进程必须掌握全局轮次，子进程每次只完成一个 backend/size 的一个测量轮。

| 全局轮次 | 同一尺寸内的顺序 | 尺寸区块顺序 |
| --- | --- | --- |
| 0 | Torch → OpenVINO → ONNX | 640 → 960 |
| 1 | OpenVINO → ONNX → Torch | 960 → 640 |
| 2 | ONNX → Torch → OpenVINO | 640 → 960 |

- 共 18 个测量子任务，串行、新进程；各配置在同尺寸内分别处于首/中/末位一次。
- 每子任务包含两个 include_detections 模式，每模式 pipeline warmup=5 + 100 张正式测量；API warmup=5 + 100 张 sequential 请求。两种模式的轮内顺序采用预声明的 false/true、true/false、false/true，所有配置相同。
- 所有正式轮次完成后，另串行执行六个配置的并发探测；每模式 API warmup=5，1/2/4 客户端各 20 请求/客户端。因此压力不会插入正式轮次之间扰动下一候选。
- 每配置正式 pipeline 行数为 `2×3×100=600`，API sequential 为 600；API warmup 为 `2×5×(3+1)=40`，concurrency 为 `2×(1+2+4)×20=280`，API 总行数 **920**。旧 890 行公式不再适用。pipeline warmup 独立计数，不进入 600 个正式请求。
- 轮次 P95 只使用该模式的正式请求，三轮 P95 取中位数；cold load、warmup、压力请求、503 分开统计，不混入正式延迟。所有排除必须有计数和原因。

### 实施

1. 抽取可测试的 schedule builder 和聚合器（拟新增，可留在比较工具内部）；运行前把完整日程、各输入 SHA、预算和工具版本写入新 run，禁止根据已测速度重排。
2. 将 `_worker` 拆为单轮 measurement 与 concurrency 两种工作单元；保留 `compare` 的公开 CLI。所有新增内部字段，如 `global_round`、`task_kind`、`task_id`，先实现再使用。
3. 以 `(config_id, mode, task_kind, global_round, sample_id/request_index)` 建立预期任务集合，检查缺失、重复、未知样本、配置 SHA 不同、失败子任务。仅总行数相等不足以证明完成。
4. `_stats`、`_api_completeness`、`adoption_completeness` 均从日程和配置推导，不依赖硬编码 range(3)、300、890。新统计 schema 升版，旧结果保持原样且标明 legacy；不要无声改变旧字段含义。
5. 父进程累计每配置的所有子任务 wall time，剩余预算传给下一任务；每配置≤900 秒、总计≤5400 秒，包括启动、校验、加载和 warmup。不能每一轮重新赠送 900 秒。
6. 冷启动/编译及真实 provider、线程、shape 每子任务记录；单轮完成行及时落盘。超时/进程失败让对应任务 partial/failed，不进行采用判断。
7. 每配置第一轮固定为跨后端 parity 对照；后两轮检查同配置输出稳定性。保留所有差异和原 exact-feature gate，不把之前诊断用的 5e-7 容差改成通过标准。
8. 同尺寸 square Torch 是后端参考；生产 rectangular 640 仍单独标明。遵守原性能、失败率、RSS 门槛，不为了重测改变它们。
9. 对 v0.3.0 与共享解码版的 profiler 复验，同样预声明 AB/BA/AB 的三轮执行顺序，使用同一冻结 100 张清单、相同模式/线程。可用脚本编排现有单轮配置；脚本和 SHA 随 run 保存，不能手工挑最好一轮。并发压力独立执行。

### 必须新增的回归

- schedule 可重复，18 个正式任务唯一且每配置三轮；每尺寸每后端的首/中/末位均衡。
- 用 fake worker 记录实际启动序列，确认是一轮后换候选，不是 worker 内藏三轮；不存在并行推理。
- 完整账本生成 600 pipeline、600 sequential API、40 API warmup、280 API 压力行；缺失一行再复制另一行补总数仍失败。
- 任一任务超时、重复 global_round、混入不同清单 SHA、后端 fallback、顺序 API 503/processing failure：不可通过完整性/采用门槛。
- warmup/压力极端耗时不影响正式轮次 P95；失败分母仍保留。
- 累计预算不会因分轮重置；剩余预算不足时停止并保留结果。纯测试不启动真实模型。

**模式来源：** `_worker`、`_profile_api`、`_concurrency_probe`、`_stats`、`_api_completeness`、`adoption_completeness`，以及 `tests/test_compare_cpu_backends.py` 的计数和门槛测试。R3/R4 修改 profiler，R5 接入时由同一集成人员串行合并。

## 9. R6：统一验证、重测与文档收口

### 验证顺序

1. 完成 F1–F5 的 red/green 回归、工具 lint 和编译后，再运行一次完整 Python 回归。覆盖率保持≥80%，当前 CI 运行时/API 专项≥85%。不为修工具而大范围改产品测试。
2. 对源码外 wheel、doctor、真实 YOLO smoke 做验收；默认安装不引入 OpenVINO/ONNX。前端未修改时复用本次审查的 12 tests/lint/build 证据，但涉及 API/前端或重新发布构建时重跑相应检查。
3. 在后端隔离环境核验实际锁定版本。补充可提交的实验依赖清单/安装说明和无本机路径的配置模板，避免重测工具只能依赖忽略目录里的环境说明运行；默认锁文件保持。
4. 校验并复用原始 .pt、四份派生资产和固定 100 张 development 清单；有文件变化时停止并说明，不能悄悄重新抽样或更换权重。
5. 仅在上述检查通过后执行一次预声明的完整重测：R4 的完整分段与 AB/BA 复验、R5 的轮换后端矩阵。不得并行启动另一份性能实验，不终止用户服务来“改善”结果。
6. 后端没有达到门槛仍记 not_adopted；本次目标是可信工具和证据，不是保证跑出加速。预算不足则 measurement_revalidated=blocked/partial，不放宽门槛补过。
7. 若执行会话已有远端提交/CI 授权，按仓库现有流程验证准确提交上的 Windows/Linux × Python 3.11/3.12；否则完成本地检查和 CI 准备，明确矩阵仍未验证，不冒称跨平台完成。

### 可运行命令模板

以下是当前已有 CLI；从仓库根运行。每条原生命令检查 `$LASTEXITCODE`，失败即处理，不只看最后一条命令状态。

```powershell
Set-Location 'D:\Codes\SafeSight'
$safeSightPython = @('.\.venv\python.exe', '.\.venv\Scripts\python.exe') |
    Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $safeSightPython) { throw '请先确认已锁定的 Python 3.11 CPU 环境' }

# 定向回归：若新增 test_backend_artifacts.py，在 R2 完成后加入此列表。
& $safeSightPython -m pytest tests/test_audit_independent_holdout.py tests/test_compare_cpu_backends.py tests/test_profile_pipeline.py --no-cov
& $safeSightPython -m ruff check src tests tools
& $safeSightPython -m compileall -q src tests tools
& $safeSightPython -m pip check
& $safeSightPython -m pytest --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=80
& $safeSightPython -m traffic_risk.cli doctor --config local/config.json
& $safeSightPython -m traffic_risk.cli smoke-rules --model-dir models
git diff --check
```

运行时/API 覆盖率专项复制执行时 `.github/workflows/ci.yml` 的真实命令。后端比较在其锁定隔离环境运行，不使用默认 `.venv` 强行安装实验依赖。

```text
python tools/audit_independent_holdout.py --candidate-manifest <jsonl> --history-manifest <jsonl> --provenance <json> --dataset-root <root> --output <new-directory>
python tools/profile_pipeline.py --config <local-frozen-json> --output <new-run-directory>
python tools/compare_cpu_backends.py --config <local-frozen-json> --output <new-run-directory>
```

配置字段以修复后 `--help` 和模板为准。数据审计退出 3 代表数据门槛未满足，可以是预期拒绝结果；不能因此修改真实 provenance 来让退出码变零。

### 文档及 gate 交付

- 新产物写 `results/cpu-repair/<run-id>/` 与 `logs/cpu-repair/`，确认被 Git 忽略；原始图片、manifest、私有路径不提交。
- 更新 `CPU_PERFORMANCE_REPORT.md`：保留旧观测值，说明旧分段缺失和非交替顺序限制；新测量单列，不覆盖旧数字。缺失段不得补造数据。
- 更新 `CPU_BACKEND_EXPERIMENT.md`：列新 schedule、schema、920 行分母、每轮结果及所有 gate；旧矩阵作为历史探索性结果保存。
- 更新 `INDEPENDENT_EVALUATION_PROTOCOL.md` 的候选内部去重规则；真实来源/人工复核状态不变。
- 更新 `CPU_OPTIMIZATION_STATUS.md` 并完成新的 repair status；区分“工具修复完成”“实验完整”“候选不采用”和“外部证据阻塞”。
- 生成新 `gate_summary.json`：至少有 F1–F5、code_fixes_verified、measurement_revalidated、default_reproducibility、platform_matrix、backend_parity/adoption、source_provenance、independent_holdout、human_review。每项附对应测试/结果 SHA 和 pass/fail/blocked/not_applicable。
- 最终交接包含修复文件、red/green 证据、检查命令退出码、真实重测 run、未完成门槛及具体下一步。不要求本轮把人工证据或质量晋级也写成通过。

## 10. 可直接交给执行 agent 的启动指令

```text
请执行 docs/CPU_OPTIMIZATION_REPAIR_AGENT_PLAN.md，修复审查确认的 F1–F5。
先核验 Git/环境与未提交工作，建立 CPU_OPTIMIZATION_REPAIR_STATUS.md。
逐项先补可失败回归，再实现修复；保留所有用户改动和历史实验。
优先修候选集内部重复审计、OpenVINO 实际资产绑定，再修 partial、插桩和轮换调度。
不要直接把普通 XML 路径传给当前 YOLO 外层 loader；按锁定版本的真实接口处理。
CPU 实验串行，代码测试全部通过后仅执行一次正式重测；预算不足如实记 partial。
默认模型、风险语义、阈值、依赖和模型晋级门槛保持不变。
缺少人工证据或平台矩阵时保留对应 blocked，继续可独立完成的工程工作。
最终提供代码修复、测量复验和发布就绪三个独立结论及证据，不强行晋级后端。
```
