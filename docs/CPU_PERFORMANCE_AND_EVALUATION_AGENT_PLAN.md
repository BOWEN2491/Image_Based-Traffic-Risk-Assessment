# SafeSight v0.3.0 后续优化：Agent 执行计划

> 状态：待执行。本文只定义后续工作，不表示已实现、已测得加速或已通过模型晋级。
>
> 编写基线：`93ba16cb6c8d46564280b3459cdec731dcd2ef9f`；已发布代码 `v0.3.0`，公开 YOLO11n 资产仍来自 `v0.2.1`。
>
> 工作目录：`D:\Codes\SafeSight`。保留旧计划 `docs/NEXT_ITERATION_AGENT_PLAN.md` 和历史状态 `docs/NEXT_ITERATION_STATUS.md`，不覆盖、不重新解释其历史实验。

## 1. 目标与执行边界

本轮优先完成可验证的 CPU 工程优化，同时补齐以后判断模型效果所需的数据证据。交付分为三类，必须分别验收：

| 类型 | 目标 | 能作出的声明 |
| --- | --- | --- |
| A：共享解码与性能测量 | 相同输入和默认模型，减少重复工作，保持结果 | 在明确环境和样本上测得多少延迟变化 |
| B：可选 CPU 推理后端 | 同一 YOLO11n、同一输入尺寸，比较 PyTorch、OpenVINO、ONNX Runtime | 在通过一致性检查的平台上，后端是否更快 |
| C：检测配置与规则评估 | 用独立新样本验证 960 等候选，以及规则输出的可解释性 | 独立数据上的检测指标、规则前提符合率；不能称为真实事故风险准确率 |

默认选择：先 A，再 B；数据准备和人工复核可与工程工作交错进行。C 的正式结论依赖独立数据门槛，不能因工程阶段结束而自动通过。

- 继续支持 Windows/Linux、Python 3.11/3.12、CPU；保留现有包名、CLI、API 成功响应、`RISK_MODE`、`high/unknown` 语义。
- 初始默认仍是 PyTorch + YOLO11n + 640 + confidence 0.25。禁止把未触发规则的 `unknown` 改为 `low`。
- 本轮不重训检测器、不恢复旧 XGBoost 历史指标、不新增 CNN 训练、不做 INT8/FP16、外部 GPU 或付费实验。
- 不升级网络型号来代替本轮优化；不把 YOLO11n 的后端加速包装成检测准确率提升。
- 数据原目录、桌面旧工程和 OneDrive 旧目录只读。新代码、环境、缓存、导出权重、日志都留在本 workspace 下。
- 本计划被要求执行时，可完成必要实现、测试和本地产物；远端 PR、合并、打标签和 Release 遵循执行会话中已有的明确授权。本文本身不代表发布已获授权或已经完成。

## 2. 已知事实与本轮需要解决的缺口

事实来源：[上一轮状态](NEXT_ITERATION_STATUS.md)、[S5 实验](S5_CPU_EXPERIMENT.md)、[错误分析](DETECTION_ERROR_ANALYSIS.md)。

- v0.3.0 发布记录为 188 个 Python 测试、84.32% 覆盖率、12 个前端测试，以及源码外 wheel、真实 YOLO smoke 验证。它们是历史证据，执行时须对当前提交重新确认相关检查。
- S5 使用 200 个 development ride groups，其中优先选择 150 张 small/tiny 灯图片和 50 张无灯图片，是压力样本，不代表自然道路分布。
- S5 B0（640/.25）mAP50–95 为 14.17%，交通灯召回 7.08%，detector-path P95 为 46.09 ms；C1（960/.25）为 15.86%、13.48%、76.15 ms。C1 值得继续研究，但其 detector-path P95 已为 1.65 倍，尚无完整 pipeline 的晋级证据。
- 当前 runtime 显式 `cv2.imread`，再将路径交给 YOLO；特征构建又为尺寸读图；可选框输出还用 PIL 打开并处理 EXIF。内部实际解码次数和耗时需要测量，不能把静态调用数直接写成加速比例。
- `tools/benchmark_latency.py` 已测 pipeline，但取旧 holdout 前 50 张、按模型分组执行多轮，不能作为新的独立测试集或交替配对测量的完整实现。
- `audit_cross_manifest()` 将跨角色重叠判为 conflict，但同角色历史 holdout 重复只记 overlap。现有 `--compare-manifest` 不足以证明“新 holdout 从未使用”。
- `source-verified` 和 `annotation_review=reviewed` 依赖人工证据。字段格式合法不等于来源已经核实。

## 3. 执行纪律、依赖与交接

按 O0 → O1 → O2 推进；O3 的自动化准备随后尽早完成。等待 O3 人工证据时可执行 O4 和 O5 的工具部分，最后统一进入 O6。禁止等待人工时反复重跑旧 holdout。

| 阶段 | 优先级 | 依赖 | 主要交付 |
| --- | --- | --- | --- |
| O0 文档、环境、接口核验 | P0 | 无 | 基线记录、允许使用的接口、运行状态文档 |
| O1 完整链路测量 | P0 | O0 | 可重复 profiler、固定性能清单、基线报告 |
| O2 共享解码 | P0 | O1 | 保持兼容的单次推理解码路径及一致性证据 |
| O3 独立数据与复核 | P1 | O0 | 历史暴露清单、独立性审计、人工复核队列 |
| O4 CPU 后端比较 | P1 | O1/O2 | 可校验导出资产、可选后端实验、配对报告 |
| O5 规则效果与候选验证 | P1 | O3/O4；工具可先做 | 规则评估、冻结候选的最终比较或明确阻塞记录 |
| O6 验收与发布准备 | P0 | 对应工程阶段 | 回归、干净安装、文档、逐项 gate 结论 |

执行 agent 应新建 `docs/CPU_OPTIMIZATION_STATUS.md`，每阶段更新：`pending / running / verified / blocked / not_adopted`、提交 SHA、命令、退出码、证据路径、下一步。`not_adopted` 适用于实验完成但收益不足；`blocked` 适用于具体证据缺失，不得冒充已验收。

- 每个阶段保持小范围改动；不覆盖用户的未提交文件、不 reset/clean 旧目录、不任意终止用户服务。
- 开分支使用 `codex/` 前缀；已经在合适分支时继续。Git 操作前检查 status、分支和 worktree。
- 有多 agent 能力时，可分别委派接口核验、数据审计、测试审查；主 agent 负责合并设计。共享源码文件只能有一个写入者，真实 CPU 推理必须串行。
- 若只有一个 agent，顺序执行即可；阶段结束必须留下足够信息让新会话继续，不依赖聊天记忆。
- 统一 seed=42、batch=1、初始 CPU threads=2；同时记录实际后端线程/流配置，不能只设置 Torch 就假定其他后端也是两线程。
- 输出写入 `results/cpu-optimization/<run-id>/`、`logs/cpu-optimization/`、`local/cpu-optimization/`。不得覆盖旧 run。
- 新环境优先放 `local/envs/`，派生模型放 `models/derived/`；验证这些目录确实被 Git 忽略。公开文档只提交汇总、配置模板和可公开测试夹具。
- 普通性能轮次最多 30 分钟；数据准备最多扫描 10,000 张/30 分钟；后端矩阵每配置最多 15 分钟、总推理最多 90 分钟；导出/编译另限每后端 15 分钟。超限保存 partial、计入未完成，不能拿缩小分母后的结果晋级。

## 4. O0：文档与接口发现，建立可复查基线

### 实施

1. 读适用的 `AGENTS.md`、现有项目文档和上轮状态。核验当前 HEAD、tag `v0.3.0`、工作区差异；记录计划基线与执行基线的差别。
2. 核验 Python、依赖锁、CPU 型号、RAM、线程数、Node、模型 manifest 和哈希。检查当前进程，不沿用历史文档中的 PID。
3. 保留可运行的 v0.3.0 对照：优先用 workspace 下隔离环境中的公开 wheel，必要时用 `local/worktrees/` 下的 detached worktree。原环境及权重不被覆盖。
4. 建立本轮 baseline JSON：源码/安装包 SHA、锁文件 SHA、模型 SHA、配置、系统、CPU、后端库版本、环境路径、样本清单 SHA。实际路径只写本地忽略文件。
5. 阅读下表源代码并在状态文档记录“已核实接口”。行号仅为本计划基线的定位提示，执行时重新定位。

| 来源 | 已有接口/模式 | 用途及限制 |
| --- | --- | --- |
| `src/traffic_risk/model_runtime.py:194` | `ModelRuntime.detect(image_path)` | 返回 BDD 风格 frames/objects；当前传路径给 YOLO |
| `src/traffic_risk/pipeline.py:28` | `run_single_image(path, runtime, *, include_detections=False)` | 保留此公开调用方式；失败不能变成正常 abstention |
| `src/traffic_risk/build_features.py` | `compute_features(objects, height, width, *, center_width_pct=0.33, near_thresh=0.05)` | 复用既有特征算法，避免为尺寸再次读图 |
| `src/traffic_risk/app.py:102,185` | `_validate_image`、`POST /api/predict` | 保留上传校验、semaphore、线程 worker、finally 清理 |
| `src/traffic_risk/inference_outcome.py`、`evaluate_detection.py` | `is_processing_failure(reason_code)`；评估层 `pipeline_failed(response)` | 复用，不能把所有 unknown 当失败或全部当成功 |
| `src/traffic_risk/prepare_evaluation.py:124,291,453` | `audit_cross_manifest`、`prepare`、CLI | 原审计不是严格历史暴露排除器 |
| `src/traffic_risk/contracts.py` | `AssetRecord`、`ModelManifest`、安全相对路径校验 | 原 `schema_version=2.0.0` 是特征契约，不能挪作后端版本 |
| `src/traffic_risk/hard_rules.py` | `apply_hard_rules(row) -> tuple[risk, reason]` | 首个触发规则决定返回；保留顺序、阈值、文本 |
| `tools/run_s5_cpu_experiment.py` | 有界执行、hash、逐图 JSONL | 复用追溯方式，不复用其样本来冒充最终 holdout |

外部文档只作为能力与接口依据：[Ultralytics OpenVINO](https://docs.ultralytics.com/integrations/openvino/)、[Ultralytics ONNX](https://docs.ultralytics.com/integrations/onnx/)。文档确认相应导出入口；当前网页示例可能使用更新的 YOLO。执行 agent 必须核对本仓库锁定 Ultralytics 的源码和参数，不能照搬最新网页的额外参数或自动升级网络。具体性能需要本机实验。

### 验证与防错

- [ ] 基线能加载已有公开模型，doctor、真实模型 smoke 通过；失败先修复环境，记录实际原因。
- [ ] 允许使用的现有接口与拟新增接口明确分列；不存在的命令不能当成已安装功能。
- [ ] 模型验证走正式 loader；禁止直接赋值 `runtime.yolo = YOLO(path)` 来跳过资产校验。
- [ ] 现有默认依赖锁不因实验准备被整体升级。

## 5. O1：完整链路测速与性能基线

### 实施

1. 新建 `tools/profile_pipeline.py`，复用现有 benchmark 的逐图记录和结果分类，新增固定配置/manifest 输入、有界执行、拒绝覆盖、阶段计时和可比较 summary。
2. 从已暴露的 development 数据中固定 100 张性能样本：覆盖不同分辨率、目标密度、交通灯、行人、车辆、无支持目标。保存身份、group、文件 SHA、尺寸和选择规则；不足 100 张如实记录。性能集允许重复使用，但不能再加入最终质量 holdout。
3. 区分三个测量入口，不混用名称：
   - detector：模型预处理、forward、NMS/映射；单独说明是否含解码。
   - pipeline：`run_single_image` 入口到返回，含解码、检测、特征、规则及可选框组织。
   - API：本地 HTTP 客户端发请求到完整响应读取；服务端另记读取/校验、临时文件写入、排队、worker、序列化/响应发送的边界。
4. 使用单调高精度时钟。保存每请求的阶段区间和父子关系；内外层不能重复相加。P95 直接从完整逐请求时长计算，禁止相加各阶段 P95。
5. 插桩默认关闭，不新增公开响应字段、不记录原图内容或真实绝对路径。若使用 ASGI wrapper 计时，明确包含到哪个 `http.response.body` 事件；客户端时间与服务端时间分别报告。
6. 对 `include_detections=false/true` 分别测量。主要门槛用网页使用的 `true`；同时保证默认 `false` 无明显退化。
7. 每个配置独立进程加载，冷启动/导出/编译时间单列；每轮 warmup=5 后测固定清单，共 3 轮。两方案用 AB/BA/AB，多方案固定 seed 的轮换顺序，防止总把某一方案放在热机后。
8. 记录逐图耗时、各轮 P50/P95、三轮 P95 中位数、峰值进程 RSS、失败数量/总数、reason_code、样本/代码/配置/模型哈希。说明 RSS 采样间隔及采样峰值的局限。
9. API 单请求延迟和并发压力分开。并发测试 1/2/4 客户端，各至少 20 请求；分别报告成功、503、其他失败、排队等待和延迟。不能把快速拒绝的 503 混入成功请求 P95。
10. 旧版无法提供细阶段插桩时，只比较可测的总时间；可以给旧版加经过审查的只读计时补丁，但必须记录补丁 SHA，不能伪称未修改的发布包自带 profiler。

### 文档与模式

读 `tools/benchmark_latency.py`、`tools/run_s5_cpu_experiment.py`、`src/traffic_risk/app.py`、`src/traffic_risk/inference_outcome.py` 和 `evaluate_detection.py`；复用现有分类及 busy/cleanup 测试。新结果必须带 reason_code，旧报告的兼容分支不能成为新增缺字段响应通过检查的理由。

### 验证与防错

- [ ] 使用可控时钟/可控延迟验证分段、总时间、分母和超时状态；异常与正常 unknown 被正确区分。
- [ ] 两次相同配置得到相同清单与参数；耗时允许变化，结果文件不能覆盖。
- [ ] 真实运行产出 baseline JSON/CSV 和 `docs/CPU_PERFORMANCE_REPORT.md` 的基线部分。
- [ ] 不要求单元测试断言机器相关的毫秒阈值；不把共享 CI runner 的延迟作为性能晋级依据。

## 6. O2：共享已解码图像，保持结果与边界

### 实施

优先从现有 OpenCV 解码路径提取以下**拟新增**内部接口；不要直接更换 JPEG 解码器。

```python
@dataclass(frozen=True)
class DecodedImage:
    bgr: np.ndarray
    width: int
    height: int

def decode_image(image_path: str | Path) -> DecodedImage: ...
def run_decoded_image(image: DecodedImage, runtime: DecodedRuntime,
                      *, include_detections: bool = False) -> dict[str, Any]: ...
# ModelRuntime 新增：detect_image(self, image: DecodedImage) -> dict[str, Any]
```

1. 新建 `src/traffic_risk/image_io.py`，明确 uint8、H×W×3、BGR、C-contiguous、尺寸一致、EXIF 方向只应用一次的契约。读取失败转换为现有 `PerceptionError`，仍由管线生成 `perception_error`。
2. `ModelRuntime.detect(path)` 保留薄适配器；生产管线解码后调用 `detect_image`，YOLO 使用 ndarray，CNN 在同一数组裁剪。用探针确认 YOLO 不再经路径触发第二次完整像素解码。
3. `run_single_image` 保留签名，新增显式 decoded runtime 协议和受测适配：支持新协议走共享路径；仅实现旧 `Runtime.detect(path)` 的外部调用者保留 legacy 路径。不得用宽泛异常捕获在生产新路径中静默退回旧路径。
4. 提取共同的“detections → features → rules/model → response”函数，避免新旧路径各复制一套规则和异常处理。
5. 用 `compute_features(objects, height, width)` 构建特征；可选框输出接收同一尺寸，保留原字段、归一化、confidence/坐标有限性检查和 300 框上限。
6. 第一轮保留 API 上传校验及临时文件流程；完整像素解码仍在 semaphore 内的 worker 执行。PIL `verify()` 是校验环节，不能为追求“一次”而删除，也不能直接算作完整解码。
7. 每请求独立持有数组，无全局图像缓存。`frozen=True` 不会冻结 ndarray；检查后端是否修改输入，必要时在明确边界复制并计入耗时，测试连续/并发请求不污染。
8. 移除临时文件或增加 bytes 入口暂不实施；只有 O1 证明磁盘写入为主要瓶颈时再提出单独变更，避免扩展本阶段。

### 验证

- [ ] 新增 EXIF 1–8 测试，覆盖 JPEG/PNG/WebP、灰度/RGB/RGBA、竖图、透明图、损坏图；使用可公开生成的测试夹具。
- [ ] 比较旧解码/模型输入与新输入的尺寸、像素、预处理张量和真实推理输出；若发现旧 EXIF 行为错误，单列为修复，不能默认为“完全等价”。
- [ ] 同一固定 CPU/权重下，固定清单的 `status/risk/reason_code/reason/features` 一致；框和分数的数值误差逐项记录。无解释的输出变化阻止按等价优化发布。
- [ ] 新路径完整像素解码计数为 1；模型预测次数不因开框而增加；legacy 适配另行标明，不计入单解码承诺。
- [ ] 运行 `tests/test_predict_pipeline.py`、`test_public_rules_detection.py`、`test_app.py`、`test_build_features.py`，保留上传拒绝、503、关闭上传、临时文件清理、无 CNN 灯框等回归。
- [ ] 对照 O1 重新测量：三轮 P95 中位数不比 v0.3.0 差超过 5%；减少解码是结构性结果，实际加速不足时直接报告，不编造收益。

模式来源：`test_optional_detection_details_use_exif_oriented_dimensions_once` 当前仅覆盖 EXIF=6 的尺寸，不证明解码次数；应扩展而不是引用名称作为完成证据。公开 API 不添加 profiler 字段。

## 7. O3：建立严格的历史暴露审计和新 holdout

### 实施

1. 汇总所有已知训练、development、旧 holdout、错误分析、人工看图及本轮性能样本清单。记录每份 manifest 的 SHA、用途、时间、覆盖范围和缺失记录；无法收集完整历史时标为 `history_coverage=incomplete`。
2. 新建 `tools/audit_independent_holdout.py`，复用现有稳定身份和像素哈希，增加严格的“历史暴露排除”语义：`sample_id`、`group_id`、`pixel_sha256`、`image_sha256` 任一与任何历史角色重叠即排除，包含 same-role holdout。
3. 原 `audit_cross_manifest()` 保持其已公开语义；新工具单独报告 `no_known_overlap`、`history_coverage`、`grouping_verified`、`source_verified`、`review_complete`、`holdout_eligible`。禁止用一个布尔值掩盖缺项。
4. 同一 ride/video/source group 不能跨开发与最终集合。`image-id-fallback` 不能证明组级独立，正式集合须排除或由人工证据补全来源映射。命名空间必须包含数据来源，防止不同数据源同名 ID 被错误解释。
5. 文件哈希/像素哈希只查精确重复；结合组信息查相邻帧，必要时生成感知近似重复的人工审查队列。来源不明不能通过“哈希不同”洗成独立。
6. 先在未看候选输出的前提下冻结抽样协议：目标 500 张、最多 500 张自然抽样新图；按来源组选择，保留原分布，不为了凑类别再向自然集加困难图。另设最多 200 张 stress 集，覆盖小灯、夜间、遮挡、黄色灯等，分别出表。
7. 正式质量晋级最低要求：自然集 500 张、至少 100 个可靠来源组；person/car/traffic light 各至少 100 个标注实例且分布于至少 25 组。该下限只是本轮操作门槛，不保证统计功效；区间仍宽时结论为证据不足。任一短缺如实报告，不用旧样本回填。
8. 人工核验 provider、原始归档/获取记录、转换链、原图和标注坐标语义；按现有 provenance 字段保存 reviewer、日期、scope、证据引用。只有真实提供的证据才能填 `source-verified`。
9. 核验标注完整性、关键类别、损坏/重复图及 EXIF 与坐标一致性。runtime 的 EXIF 优化不能顺手改变数据标注坐标或历史 pixel hash。
10. 人工复核者可以看图和标签，但最终模型候选选择者在冻结前不能看最终集预测结果。保存接触角色和时间；若同一人同时调参并审阅最终图像，披露限制，必要时换新来源组。
11. 来源证据、人工复核未完成时，工具和队列可以验收；数据门槛保持未通过。agent 不得代签人类复核，不得将自动检测结果当人工真值。

### 现有接口与新增工具

已有命令 `traffic-risk prepare-evaluation` 支持 `--config`、`--output`、`--seed`、`--max-images`、`--max-crops`、`--max-train-images`、`--max-seconds`、`--provenance`、可重复的 `--compare-manifest`。可复用它生成有界清单，但其旧 holdout 角色必须经新审计后才具有新含义。

新工具至少接收候选 manifest、可重复的历史 manifest、来源记录、输出目录；输出 `eligible.jsonl`、`excluded.jsonl`、`audit.json`。严格检查文件变更、输入/输出路径和缺失身份。不能只传一个旧 manifest 就宣称历史覆盖完整。

### 验证与防错

- [ ] same-role 旧 holdout、跨角色 ID、同组、同像素不同文件、同文件不同 ID 均被正确排除。
- [ ] 缺少可靠 group 或历史记录时不能通过正式门槛；来源字段校验与实质证据分开。
- [ ] 相同 seed/输入产生相同清单和 SHA；保留拒绝覆盖、拒绝写源数据目录、有界扫描测试。
- [ ] 扩展 `tests/test_prepare_evaluation.py`；新工具的单元测试用合成元数据，不能依赖私人图片。
- [ ] `docs/INDEPENDENT_EVALUATION_PROTOCOL.md` 写清人类待办、数量短缺、预训练模型数据暴露未知等限制，不承诺超出已知历史的绝对独立。

## 8. O4：同权重 CPU 后端实验

### 实施

1. 在隔离实验环境中先试 OpenVINO，再试 ONNX Runtime CPU；与 PyTorch 共用原始 YOLO11n checkpoint、类别映射和公共前后处理。禁止为安装后端破坏默认 CPU 环境。
2. 核验锁定 Ultralytics 版本支持的 `YOLO(...).export(format="openvino")` / `format="onnx"`。额外的 shape、dynamic、opset、simplify、NMS 参数逐项查该版本实现，记录采用值。预装锁定依赖，禁止运行时自动 pip 安装/隐式下载。
3. 首轮仅 FP32、CPU、batch=1。验证导出权重精度及实际执行精度，不把 `half=False` 当成所有后端都未压缩的充分证据。无法确认时记录实际精度，不进入 FP32 等价结论。
4. 优先用 Ultralytics 支持的后端加载方式复用映射；若为控制线程或动态尺寸必须使用原生 runtime，写最小 adapter，所有预处理、NMS、坐标还原必须与参考实现比较。
5. 特别核验 `rect`/letterbox/stride、输入张量实际 H×W、插值、颜色转换和 NMS 是否一致。640 这个参数相同不保证实际张量相同。静态方形后端若无法匹配原路径，建立相同方形 PyTorch 对照，并把“生产预处理改变”另列为行为变化，禁止混称纯后端加速。
6. 首轮矩阵最多 6 配置：PyTorch/OpenVINO/ONNX Runtime × 640/960；confidence=.25、NMS IoU=.45、max_det=300，所有配置均使用共享解码路径。旧 C2/ROI 不重新扩展，除非本轮证据指出明确必要性。
7. 先完成同尺寸一致性检查，再按 O1 测性能。AP 评估另用既有 .001 阈值导出预测，操作召回/精度用 .25，不能混用 AP 阈值和产品阈值的延迟。
8. 保存源 `.pt` SHA、导出环境/源码、输入 shape/dtype/layout、预后处理、类别映射、导出参数及每个输出文件大小/SHA。使用独立 `backend_artifact_manifest.json`，不改旧 `ModelManifest.schema_version`，不向 models 模式严格六文件集合塞额外资产。
9. 派生 manifest 应验证 bundle 相对路径、路径越界、重复条目、必需文件、大小与哈希、源权重身份；`.xml/.bin/metadata` 等实际需要的全部文件均纳入。原始公开资产仍先走正式验证。
10. 显式指定的后端缺依赖、资产损坏或不支持时明确失败；实验不允许静默回退 Torch。结果必须记录实际 provider/device/backend，不能只打印请求的后端名。
11. 只有通过下述 B 门槛的 640 后端才值得接入公开可选配置。通过时新增显式配置（建议 `RISK_DETECTOR_BACKEND` 和派生 manifest 路径），默认 Torch；loader/doctor/readiness/CLI/API 共用。变量名、合法值、缺依赖修复命令写入 `.env.example` 和文档。
12. 后端依赖作为独立 optional extra 和锁文件提供；不进入默认 CPU 安装。即使本机 Intel 测得提升，也不自动替换所有平台默认。960 保持实验配置，直到 C 门槛通过。

### 同尺寸一致性门槛 B

- 在固定性能/开发一致性集上按类别一对一匹配：以 IoU≥0.99、score 绝对差≤0.001 为初始容差；所有未匹配框、临界 confidence/NMS 变化单列，禁止先过滤掉差异样本。
- 检查全部输入的 `status/risk/reason_code`、首个规则和关键特征变化。不能有未解释的 `high↔unknown` 变化。存在行为变化则转入 C 门槛，不能按纯性能替换发布。
- 全流程与 API 的三轮 P95 中位数相对同尺寸 Torch 至少改善 20%，失败不增加，峰值 RSS 不超过 1.25 倍；两种 include_detections 路径均不得退化超过 5%。20% 是本轮采用门槛，不是预期实验结果。
- 最多进行一次已记录原因的测量复核；不得反复跑到取得有利数字。记录每轮结果和波动，而不只选最好一轮。
- 达不到门槛则 `not_adopted`，保留实验报告；不为引入新依赖而降低标准。没有独立 holdout 时只声称已测样本上的一致性，不声称整体检测性能不变。

### 验证与防错

- [ ] 真实导出 → manifest → 回加载 → 推理完成，不以 mock 代替后端可用性。
- [ ] 错误 hash、缺少 `.bin`、错误类别、错误 shape、未知 backend 和路径越界在加载时失败。
- [ ] 默认安装没有后端库仍正常工作；选择缺失后端时 doctor 给出有效安装命令。
- [ ] 对将公开支持的后端跑 Windows/Linux、Python 3.11/3.12 的安装和真实 smoke；不支持的组合明确排除于该可选后端支持表，不能影响默认路径矩阵。
- [ ] 生成 `docs/CPU_BACKEND_EXPERIMENT.md`；不采用候选也是完整实验结论。

## 9. O5：规则效果检查与最终候选评估

### 工具与人工复核

1. 在 `hard_rules.py` 中复用同一组谓词，增加内部规则 trace，稳定 rule ID、触发顺序、all matches 和 first match。`apply_hard_rules` 继续返回原二元组，顺序/阈值/文本不变，禁止复制一套易漂移的统计规则。
2. 新建 `tools/evaluate_rule_behavior.py`，输出各规则命中数、重叠、high/unknown 分布、正常 abstention 原因、pipeline failures，以及 baseline→candidate 的输出转换表。
3. `unknown` 不以“越少越好”为目标；处理失败不能混入正常规则未触发。公开 rules 不加载 CNN，灯色仍为 unknown；不能声称仅检测更多灯就改善红灯规则。
4. 准备最多 200 张本地人工规则复核队列，覆盖规则命中、未命中、检测变化和困难情况；各样本标明抽样来源。选择依赖模型输出的队列属于诊断集，不能计算自然分布总体准确率。
5. 人工先看原图填写可见前提、标注质量和 `yes/no/uncertain`，再看模型理由。至少 20% 双人独立复核并记录分歧/裁决；没有第二位复核者则记录证据限制，不能由 agent 模拟两人投票。
6. 报告“可见规则前提的符合率/漏触发及支持数”，同时保留 uncertain 数量；这不是事故概率、真实驾驶风险或法规判定。单帧看不到速度/道路优先权时不能补猜。

### 冻结候选与 C 门槛

1. 仅根据 development 和性能结果选择至多一个质量候选（优先考虑 960/.25；不强制选择）。冻结源码、后端、输入尺寸、阈值、导出产物、候选理由及协议 SHA 后，才运行新 holdout。
2. O3 任一证据门槛未满足，正式比较保持 blocked；可完成工具、开发实验和 B 类可选后端验证，不触碰最终标签来调参。
3. 最终自然集一次性比较 baseline 和冻结候选，stress 集独立出表。保留所有失败样本和计数，失败图中的真值目标计入召回分母，不能通过跳过难图提高指标。
4. 沿用仓库已实现的 101-point AP/IoU .50:.05:.95 定义，明确不等同完整 COCO area/crowd 指标；同时报告逐类 precision/recall、small/tiny 分层及支持数。
5. 在来源组层面配对 bootstrap：目标 1,000 次，所有候选使用相同重采样组。计算总体 mAP 差及 person/car/traffic light 召回差的区间；重采样必须正确保留组被抽中多次的权重。预算最多 30 分钟，未完成不得冒称有 CI。
6. 晋级条件全部满足：mAP50–95 点估计提高≥1 个百分点，差值 95% 区间下界>0；person/car/traffic light 每类召回下降≤1 个百分点，且差值区间下界≥−1 个百分点；失败率不增加。
7. 完整 pipeline 与 API 的 P95 均≤已发布 v0.3.0 同机同清单基线的 1.5 倍；另报告与 O2 优化后 640 基线的比例，防止隐藏机会成本。不能用 detector-only 时间替代。
8. `high↔unknown` 和规则触发变化完成审查，无未解释的错误新增；检测指标通过仍不能直接声称风险判断有效性提高。
9. 任何门槛失败、样本不足、区间过宽或人工证据缺失，结论为“不晋级”，默认继续 640。看过最终结果后该 holdout 标记为已暴露；如需再调候选，必须启动新一轮协议和新 holdout。

### 验证与防错

- [ ] 使用规则阈值边界、多个规则同时满足的样本测试 trace 与原返回完全一致。
- [ ] 规则报告能区分自然集、压力集、定向复核集，分母和 uncertain 明确。
- [ ] 通过合成分组样本验证 bootstrap、失败分母、候选冻结和门槛汇总；不使用看过的最终集来修测试期望。
- [ ] 写入 `docs/RULE_BEHAVIOR_REPORT.md` 和最终比较报告；人工缺项列具体证据，不笼统写“全部完成”。

## 10. O6：整体验收与发布准备

### 必须保留的回归

- Windows/Linux × Python 3.11/3.12 默认环境：锁定安装、`pip check`、编译、lint、完整 Python 回归、wheel 源码外安装。
- 维持全项目 coverage≥80%，运行时/API 门槛≥85%；新增可选模块纳入有意义测试，不为过门槛大范围排除代码。
- 前端 lint、12 个既有测试及新增必要测试、生产 build；浏览器真实上传、框开关、竖图/EXIF 对齐、换图清理、错误展示。
- 真实 YOLO：交通灯且无 CNN、行人/车辆、无支持目标；坏图片、缺失/损坏权重由专门负例覆盖。私人数据不成为公开 CI 必需文件。
- 并发、503、上传大小/像素限制、临时文件清理、离线推理仍正常；optional backend 的离线回加载另测。
- README 在新的源码外环境逐条执行，禁用旧目录隐式访问；模型下载后阻断外部网络仍可完成 CLI/API 推理。记录验证方式，不只写“应该离线”。

### 命令约定

下面是**现有命令**的 PowerShell 模板；先确认环境布局，本仓库历史使用 Conda prefix `.venv/python.exe`，不能固定假定 `Scripts/python.exe`。

```powershell
Set-Location 'D:\Codes\SafeSight'
$safeSightPython = @('.\.venv\python.exe', '.\.venv\Scripts\python.exe') |
    Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $safeSightPython) { throw '先按当前 README 建立锁定 Python 3.11 CPU 环境' }
& $safeSightPython -m pip check
& $safeSightPython -m ruff check src tests tools
& $safeSightPython -m compileall -q src tests tools
& $safeSightPython -m pytest --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=80
& $safeSightPython -m traffic_risk.cli doctor --config local/config.json
& $safeSightPython -m traffic_risk.cli smoke-rules --model-dir models
git diff --check
```

每个外部命令后检查 `$LASTEXITCODE`；不能因最后一个命令成功而忽略前面的失败。`local/config.json` 不存在时按 README 创建本地配置，不提交机器路径。运行时/API 的专项覆盖命令复制当前 `.github/workflows/ci.yml` 的真实配置。

在 `risk_frontend/` 执行 `npm ci`、`npm run lint`、`npm test -- --run`、`npm run build`。打包与源码外安装复用 `python -m build` 和 `python -m tools.wheel_smoke <wheel> --lock requirements-cpu.lock`，使用本轮明确的 wheel 文件，不能 glob 到旧包后继续验证。

以下是**必须先实现再执行的拟新增工具契约**，不存在时先完成对应阶段：

```text
python tools/profile_pipeline.py --config <local-json> --output <new-run-directory>
python tools/audit_independent_holdout.py --candidate-manifest <jsonl> --history-manifest <jsonl> --provenance <json> --output <new-directory>
python tools/export_cpu_backend.py --config <local-json> --output <new-artifact-directory>
python tools/compare_cpu_backends.py --config <frozen-local-json> --output <new-run-directory>
python tools/evaluate_rule_behavior.py --config <local-json> --output <new-run-directory>
```

其中 `--history-manifest` 可重复。每个新工具必须提供 `--help`、无绝对路径的配置模板、参数验证、明确错误退出和输入哈希记录；模板说明数据/模型路径的解析基准。配置中固定 seed、样本清单、模型身份、预算、线程、rounds、warmups、include_detections 和后端参数，不靠隐藏环境默认值。

### 最终 gate 与交付

执行 agent 生成 machine-readable `gate_summary.json`，每门使用 `pass/fail/blocked/not_applicable` 并给证据路径，至少包括：`default_reproducibility`、`decode_parity`、`performance_measurement`、`backend_parity`、`backend_adoption`、`source_provenance`、`annotation_review`、`independent_holdout`、`quality_promotion`、`rule_review`。只有本次已完成的测量可以填 pass。

| 可交付版本 | 条件 | 允许变化 |
| --- | --- | --- |
| 维护版本候选 | 默认复现、解码一致性、相关性能与安全回归通过 | 共享解码、测量工具、文档；不依赖数据晋级 |
| 可选后端版本候选 | B 门槛和所声明平台安装/真实 smoke 通过 | 用户显式选择的后端、独立资产契约及文档；默认 Torch |
| 检测配置更新候选 | O3/O5 的 C 门槛全部通过 | 已冻结并验证的配置；更新资产/模型卡/变更说明 |

实际版本号按最终变更和当时最新版本决定；不要为了完成计划强发三个版本。已完成工程可以独立交付，质量实验“不晋级”同样是有效结论。

发布准备应包含：变更说明、支持矩阵、汇总报告、资产校验、回退命令及上一个可用版本。若会话已授权正式发布，按仓库现有流程执行并核验准确 PR head 的 CI、匿名下载哈希、公开 wheel 源码外安装；未授权时保留可审阅产物和明确下一步，不声称 Release 已发布。

## 11. 可以直接交给执行 agent 的启动指令

```text
请执行 docs/CPU_PERFORMANCE_AND_EVALUATION_AGENT_PLAN.md。
先读本文、适用仓库指令、NEXT_ITERATION_STATUS.md 和当前 Git 状态，
核验执行基线，建立 CPU_OPTIMIZATION_STATUS.md；保留旧计划和用户改动。
从 O0/O1 开始，实现并验证 O2 共享解码，然后完成 O3 自动审计与复核队列。
人工证据缺失时明确列出待办，继续独立的 O4/O5 工具与开发实验。
所有真实 CPU 实验串行、有界、保留逐图证据；代码与输出都留在 D:\Codes\SafeSight。
本轮不训练检测器、不量化、不自动升级模型、不伪造人工来源/标签证据。
分别报告工程复现、行为一致性、后端收益和质量晋级，不把它们混成“全部完成”。
每阶段留好交接记录，最终完成 O6 可执行范围内的验收与发布准备。
```
