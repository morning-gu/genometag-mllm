# 方向 A: GenomeTag-MLLM — MLLM 预测 MovieLens Genome Tag 关联度评分

> 研究计划（2026-10-19 整理版：2026-10-16 真实数据核实修订与 2026-10-17 模型切换已并入正文；旧版见 git 历史）
> 目标场所：RecSys 2027 / CIKM 2027 · Novel Method（IMRaD）· ACM 格式 · 正文 6,000-8,000 words
>
> **2026-10-28 转向**：实验全部完成，结果落入 §1 反转预案（诚实负面 + 探针方法论）。目标场所改为 **SIGIR 2027（方法论优先叙事）→ CIKM 2027 级联**；本文档 §5"预期结果"已过时。成文交接以 **`docs/paper/sigir2027-handoff.md`** 为唯一权威文档。

## 1. 研究问题与核心论点

**研究问题**：MLLM 能否仅从电影多模态内容（关键帧 + 剧本 + 台词）预测出与 MovieLens genome tag 体系一致的关联度评分？

**核心论点**：多模态大语言模型能够仅从电影内容预测 genome tag 关联度，为推荐系统提供无需用户行为的可扩展自动化标签方案（输入纯内容驱动，天然冷启动可用）。

**反驳者视角**：MLLM 可能只是在"回忆"训练数据中已知的 MovieLens 标签，而非真正从内容推理；且 genome-scores 本身由 SVD++ 从用户行为训练，MLLM 预测的可能是"用户会怎么标"而非"电影内容是什么"。

**防御**（两道防线）：
1. **双记忆探针**（§4，因果检验）——title-only 盲测臂 + 错配内容交叉探针，直接测量"预测是否由内容因果驱动"。
2. **content/preference 标签分解**——分别评估内容驱动型标签（"dark atmosphere"）与偏好驱动型标签（"overrated"），后者恰是对"预测用户行为"质疑的直接检验。

**论证反转预案**：
- 若 MLLM 不敌 genre-only 基线 → genome tag 远比 genre 细粒度（1,128 vs 20），genre-only 无法区分 "dark"/"noir"/"bleak"；
- 若探针显示记忆主导 → 论文叙事转为"MLLM 标注的记忆风险分析"（诚实负面发现，同样是贡献）。

## 2. 数据

| 数据 | 角色 | 关键事实 |
|------|------|----------|
| MovieLens ml-latest genome | ground truth | 1,128 tags；relevance 长尾（mean 0.111，≥0.5 仅 3.6%） |
| MovieNet movie1K | 多模态内容 | 关键帧全量；剧本 480 部、字幕 816 部；年份 1939-2018 |
| 对齐实验集 | 评估全集 | **1,035 部**（标题+年份匹配且全 genome 覆盖） |

**设计事实**（2026-10-16 真实数据核实）：
- 年份止于 2018 → 时序分割不可承载"推理 vs 记忆"论证（由双探针替代，时序分析降级为描述性检查）；
- 冷启动子集逻辑上不可能非空（genome 由用户标签训练，无用户标签的电影必然无 ground truth）→ 冷启动转为**方法属性**而非实验（§3.2），其价值证明留给方向 D（端到端推荐）；
- 三模态全有宇宙 440 部（43%）：剧本覆盖 51% 且集中于 90s-00s（≥2015 仅 1 部）→ 消融矩阵与时序留出不可兼得，评估协议采用电影级留出（§4，协议 v3）。

## 3. 方法

**3.1 标注流程**
- **Tag 语义分组**：1,128 tags → 语义组 + content/preference 类型标注 + 一句话定义（qwen3.8-max 生成；组 ≤60 tags 以控 prompt 长度）
- **Prompt**：电影多模态块（12 关键帧 + 剧本 4,000 tok + 台词 3,000 tok）× 组内 tag 定义，输出 JSON `{tag: score(0-1)}`
- **主模型**：DashScope qwen3.7-plus（思考模式）；每任务 3 轮取平均；few-shot 2 例（仅从 <2015 训练参考池选取，防泄露）
- **工程基础**：磁盘缓存（SHA256(prompt+model+params+run)）+ 断点续跑 + 8 线程并行 + 看门狗（崩溃/卡死自动重启零损失）

**3.2 方法属性**：MLLM 输入不含任何用户行为 → 天然冷启动可用（方法/discussion 一句话 + limitation 阐明设定边界）。

**方法防御**：
- "为什么不用专门训练的小模型？" → 小模型需要大规模标注训练数据，恰是本问题要消除的依赖（鸡生蛋）；MLLM 零样本。BERT 基线即该替代方案的实证检验。
- "genome-scores 是用户行为训练的" → 正是本文要量化的分解（content/preference，§4）。
- 最大局限：1,128 tag 无法单次评分，分批引入组间差异——以语义分组 + 组内定义 + 多轮平均缓解。

## 4. 评估协议（v3：内容完备宇宙 + 电影级留出）

**宇宙与划分**：评估宇宙 = 三模态（关键帧+剧本+字幕）全有的 **440 部**；测试集 = 宇宙内按年代分层的随机 **20%（~88 部**，seed 固定，清单入 git：`data/processed/eval_test.txt`）；拟合池 = 测试集补集（**~947 部**，含部分模态电影——基线只用 genre/统计，模态可得性无要求）。**主表与消融矩阵在同一测试集上评估**（一张表：方法 = 6 个 MLLM 臂 + 4 基线）。

**指标**：RMSE / MAE / Precision@10 / NDCG@10（vs genome-scores），按 per-movie 计算 + 配对显著性检验（Wilcoxon signed-rank）

**子集视图**：全部 · tag:content · tag:preference · 年代分桶（描述性）

**消融矩阵**：full / video-only / script-only / dialogue-only / video-script / **title-only（探针 1 臂）**——宇宙保证各臂评估集同构、三模态真实可得

**记忆探针**（"推理 vs 记忆"核心论证）：
- **探针 1 · 无内容盲测**：prompt 仅标题+年份，零内容块，与消融臂并列。title-only 性能 = 非内容分量（世界知识+记忆）；full − title-only = 内容驱动净贡献。title-only ≈ full → 记忆主导；title-only ≪ full → 内容推理真实有效。
- **探针 2 · 错配内容交叉**：电影 A 标题 + 电影 B 内容，同类型/跨类型双对照，~150 对。度量 corr(pred, truth_A) vs corr(pred, truth_B) 的相对迁移：钉住 A 真值 → 忽略内容（记忆）；移向 B 真值或崩坏 → 内容被真实使用。
- **局限声明**：两探针均无法完全分离"合法世界知识"与"MovieLens 标签记忆"——从致命弱点降为可讨论的 limitation。
- 探针套件本身构成方法论贡献（MLLM 标注的内容-记忆分解探针）。

**基线**：random / 全局均值（拟合池统计）/ genre→tag 均值映射（拟合池统计）/ 剧本 + BERT 多标签回归。
**防泄露原则**：一切拟合组件（均值、genre 映射、BERT、few-shot）仅使用拟合池（测试集补集）数据。

**新片子分析（描述性）**：≥2015 的 95 部（帧+字幕自适应模态）作为部署现实展示——MovieNet 剧本覆盖不达近年电影，如实报告内容覆盖偏差。

## 5. 预期结果与叙事

**最重要发现（预期）**：MLLM 在内容驱动型标签上显著优于偏好驱动型标签，且三模态组合优于任何单模态。

**结果结构**：主实验表（MLLM vs 4 基线）→ 模态 × 标签类别贡献热力图 → 双探针分析 → 时序分桶描述性分析 → 案例分析（2-3 部典型电影）。

**初步证据**（2 部 vrfy 对照，qwen3.7-plus 思考模式，待 pilot 确认）：content NDCG 0.555（genre 0.582 / mean 0.508）；preference NDCG 0.242（genre 0.663）——不对称性已现雏形。

## 6. 章节计划

- **Introduction (800-1,000w)**：genome tag 依赖用户行为的冷启动困境（Harper & Konstan 2015；Deldjoo 2017）→ MLLM 多模态能力成熟 → Gap：无工作将 MLLM 与 genome tag 评分体系对齐（最近邻 TrUMAn/TiM 用 TV Tropes 体系、输入单一、模型非现代 MLLM）→ 贡献三句：① 首个 MLLM ↔ genome 对齐框架 ② 系统模态消融 ③ 内容-记忆分解双探针套件
- **Related Work (1,200-1,500w)**：三故事线——MovieLens 标签体系（SVD++ 起源/演化/局限）/ 电影多模态内容理解（mise-en-scene → MovieCLIP → VLM4Rec）/ MLLM × 推荐（ViLLA-MMBench、LLMRec、RAG-VisualRec）。对话点：TrUMAn (2021) 报告视频-only trope 分类仅 12% 准确率，本文检验现代 MLLM 是否颠覆该结论。
- **Methodology (2,000-2,500w)**：§3 + §4 内容
- **Results (1,500-2,000w)**：§5 结构
- **Discussion (800-1,000w)**：vs TrUMAn（能力代际差）/ vs VLM4Rec（描述生成 vs 直接评分）/ vs ViLLA-MMBench（LLM 摘要 vs MLLM 看内容）。核心记忆点：视频模态对视觉风格类标签的边际贡献 = "看懂电影"的不可替代价值。
- **Conclusion (400-500w)**：能力成立（内容型标签）+ 边界清晰（偏好型标签、记忆风险由探针量化）。

## 7. 论证压力测试

1. genome tag 依赖用户行为训练，冷启动无法生成 → 文献支撑
2. 现代 MLLM 可从内容直接推理标签 → 主实验 vs 基线
3. 三模态 > 任何单模态 → 消融矩阵（待数据）
4. 预测由内容因果驱动而非记忆 → 双探针（待数据）
5. 细粒度上超越 genre-only → 主实验（待数据）

**最弱点**：第 4 步——若探针显示记忆主导，核心论点转为"记忆风险分析"（叙事预案已备，§1）。

## 8. 关键文献定位表

| 位置 | 引用文献 | 角色 |
|------|---------|------|
| Intro 背景 | Harper & Konstan (2015) | MovieLens 数据集官方引用 |
| Intro 动机 | Deldjoo et al. (2017) arXiv:1704.06109 | 视觉特征替代人工标签的先驱 |
| Intro 技术趋势 | VLM4Rec (2026) arXiv:2603.12625 | MLLM 语义描述用于推荐 |
| Related Work 故事1 | GroupLens SVD++ | genome tag 生成方法 |
| Related Work 故事2 | MovieCLIP (2022) arXiv:2210.11065 | CLIP 弱标注电影场景 |
| Related Work 故事3 | ViLLA-MMBench (2025) arXiv:2508.04206 | LLM 丰富 MovieLens 元数据 |
| Related Work 对比 | TrUMAn (2021) arXiv:2108.04542 | 最近邻——trope 理解 |
| Related Work 对比 | TiM (2024) arXiv:2406.10923 | 最近邻——LLM trope 推理 |
| Method 数据 | MovieNet (2019) arXiv:1910.09368 | 电影多模态数据集 |
| Method 对齐 | StoryMovie (2026) arXiv:2602.21829 | 剧本-字幕-视频对齐 |
| Eval 基线 | Folksonomication (2018) arXiv:1808.04943 | 文本标签预测基线 |
| Eval 警示 | ML-1M Memorization (2026) arXiv:2601.02002 | 数据泄露警示 |

## 9. 时间线

| 阶段 | 内容 | 时点 | 产出 |
|------|------|------|------|
| ✅ 数据与框架 | 对齐 1,035 部 + 分组 + 标注管道 + 三方 vrfy 对照 | 2026-10 | 实验框架（已合入 main） |
| 主实验 | 协议 v3：6 臂 × 88 部测试集 × 3 轮（qwen3.7-plus 思考 + 8 并行，~52k 任务） | 2026-10 ~ 2026-11 | 主实验表（含消融） |
| 探针 + 分析 | 错配探针 ~150 对 + per-movie 显著性分析 | 2026-11 ~ 2026-12 | 探针分析 + 显著性检验 |
| 论文写作 | Intro / Related / Method → Results / Discussion / Conclusion | 2026-12 ~ 2027-02 | 完整初稿 |
| 内审 + 投稿 | 内部评审 + 修改打磨 | 2027-02 ~ 2027-03 | 终稿 → RecSys 2027 / CIKM 2027 |

## 10. 目标场所

RecSys 2027（ACM Conference on Recommender Systems）或 CIKM 2027
论文类型：Novel Method（IMRaD 结构）· 引用格式：ACM 会议标准 · 预估字数：6,000-8,000 words（正文）
