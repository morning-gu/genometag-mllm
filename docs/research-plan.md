# 方向 A: GenomeTag-MLLM

> academic-paper plan 模式 | 2026-10-15
> **设计修订 2026-10-16**（真实数据核实后）——见下方修订块

---

## 🔬 设计修订（2026-10-16，数据核实后）

**核实事实**（MovieNet movie1K，服务器 /cache/MovieNet）：
- 年份范围 1939–2018，**≥2019 = 0** → 原时序分割（2022 分割点）不可执行
- 1,035/1,100 部有 genome scores → 实验集 ~1,035 部（符合"~1,000 部"假设 ✓）
- 冷启动子集**逻辑上不可能非空**：genome scores 由 SVD++ 从用户标签训练，无用户标签的电影必然无 genome ground truth（设定级矛盾，非数据问题）

**修订决定**：

1. **"推理 vs 记忆"论证改用双探针组合**（替换时序分割）：
   - **探针 1 · 无内容盲测（title-only 臂）**：prompt 仅含标题+年份，零内容块，进消融矩阵与 video-only/script-only/dialogue-only/full 并列。title-only 性能 = 非内容分量（世界知识+记忆）；full − title-only = 内容驱动净贡献。title-only ≈ full → 记忆主导（诚实负面发现，叙事预案已备）；title-only ≪ full → 内容推理有真实贡献。
   - **探针 2 · 错配内容交叉探针**：电影 A 标题 + 电影 B 内容，同类型/跨类型双对照，~150 对。**因果检验**：预测钉住 A 真值 → 忽略内容（记忆）；移向 B 真值或崩坏 → 内容被真实使用。度量 corr(pred, truth_A) vs corr(pred, truth_B) 的相对迁移。
   - 组合优于时序分割：直接测量"预测是否由内容因果驱动" vs 间接代理"该片是否在训练语料中"；探针套件本身构成方法论贡献（MLLM 标注的内容-记忆分解探针）。
   - **局限声明**：两探针均无法完全分离"合法世界知识"与"MovieLens 标签记忆"——从致命弱点降为可讨论的 limitation。
2. **时序分割降级为按年代分桶的描述性分析**（代码保留 temporal_split_year，不再承载核心论证）。分割点定为 **2015**（数据止于 2018：<2015 训练参考池，≥2015 测试视图 ~95 部非空）——2022 分割点的 ≥2022 视图恒为空，已废弃。
5. **实验模型切换**（2026-10-17）：自部署 qwen3.5-4b（max_model_len 3072，内容被压缩至 4 帧/400/250 token，首跑 RMSE 显著劣于 mean/genre 基线）→ DashScope 云端 **qwen3.7-plus**；内容预算恢复设计值（12 帧 / 4000 / 3000 token）。
3. **冷启动转为方法属性**：MLLM 输入纯内容驱动（关键帧+剧本+台词，不含用户行为），天然冷启动可用——方法/discussion 一句话说明 + limitation 阐明设定边界；冷启动**价值**的实验证明留给方向 D（端到端推荐）。
4. 论文主线：**模态消融 + content/preference 标签不对称 + 记忆探针套件**。

---

 # ============================================================
 # 方向 A: GenomeTag-MLLM
 # MLLM 预测 MovieLens Genome Tag 关联度评分
 # ============================================================

 ## A-Step 0: 研究基础确认

 **已有材料**:
 - 文献综述: 已完成（TrUMAn, TiM, VLM4Rec, ViLLA-MMBench 等 15 篇核心文献定位）
 - 数据: MovieLens genome-scores.csv + genome-tags.csv 公开可用；MovieNet 数据集公开可用
 - 研究问题: 已确定——"MLLM 能否从电影多模态内容预测与 GroupLens SVD++ 一致的 genome tag 关联度？"
 - 代码基础: MovieGraph-ToM 的多模态对齐管道可参考

 **研究就绪度**: 充分，可直接进入论文规划。

 ## A-Step 1: 核心论点 (Thesis Crystallization)

 > **[INSIGHT: thesis_statement]**
 > 多模态大语言模型（MLLM）能够仅从电影内容（关键帧+剧本+台词）预测出与 MovieLens genome tag 体系一致的关联度评分，且在冷启动场景下（新电影无用户标签）显著优于随机基线，为推荐系统提供可扩展的自动化标签生成方案。

 **反驳者视角**: MLLM 可能只是在"回忆"训练数据中已知的 MovieLens 标签，而非真正从电影内容推理。且 genome-scores 本身是 SVD++ 从用户行为训练的，MLLM 预测的可能是"用户会怎么标注"而非"电影本身的内容特征"。

 **防御**: (1) **双记忆探针**（2026-10-16 修订，替换原时序分割——movie1K 年份止于 2018，时序分割不可行）——探针 1"无内容盲测"：仅给标题的 title-only 臂与全模态性能之差 = 内容驱动净贡献；探针 2"错配内容"：A 标题+B 内容的因果检验，预测若钉住 A 真值则为记忆、移向 B 真值则为内容使用。时序分析降级为按年代分桶的描述性稳健性检查。(2) 消融实验——如果 MLLM 仅用剧本（不输入视频/台词）就能达到大部分性能，说明文本信息已足够，视频模态的边际贡献量化了"真正看懂电影"的增量价值。

 ## A-Step 2: 章节计划 (Chapter Plan)

 ### A-Introduction (800-1000 words)

 **读者应感受到的紧迫感**: MovieLens 数据集自 1997 年发布以来，genome tag 的生成一直依赖大规模用户行为数据训练 SVD++ 模型。新电影上架时没有用户标签，冷启动推荐效果差。同时，人工标注成本高、一致性差。MLLM（GPT-4o/Gemini/Qwen-VL）在 2024 年后展现出强大的多模态理解能力，但尚未有人将其应用于 MovieLens 标签自动化生成。

 **研究 Gap（一句话）**: 没有发表的工作将 MLLM 的多模态理解能力与 MovieLens genome tag 评分体系对齐，现有最接近的 TrUMAn/TiM 用 TV Tropes 体系（132-200 个叙事模式标签），输入仅视频或视频+摘要，模型非现代 MLLM。

 **章节内容**:
 1. MovieLens 数据集及其 genome tag 体系的背景（引 Harper & Konstan, 2015）
 2. 自动化标注的动机：人工标注成本 + 冷启动问题（引 Deldjoo et al., 2017）
 3. MLLM 技术成熟度：2023-2024 年 MLLM 长视频理解能力突破（Technology Cycle 探针命中）
 4. 研究 Gap 和本文贡献的三句话声明

 **预期贡献声明**:
 1. 首个将 MLLM 输出与 MovieLens genome tag 关联度评分对齐的方法
 2. 系统的模态消融实验，量化视频/剧本/台词各自的标签预测贡献
 3. 冷启动场景下的时序分割评估，区分"推理"与"回忆"

 ### A-Related Work (1200-1500 words)

 **讲故事数**: 3 个故事线
 1. **MovieLens 标签体系**: genome tag 的起源（SVD++ from user tags + Wikipedia）、演化、局限
 2. **电影视觉/多模态内容理解**: 从 mise-en-scene 视觉特征（Deldjoo, 2017）→ MovieCLIP CLIP 弱标注（2022）→ VLM4Rec 语义描述生成（2026）的演进
 3. **MLLM 在推荐系统中的应用**: ViLLA-MMBench（LLM 丰富元数据）、LLMRec（图增强）、RAG-VisualRec（RAG 推荐）

 **文献综述最终结论**: 现有工作在"多模态内容理解"和"推荐系统元数据丰富"两条线上分别推进，但两条线的交汇点——用 MLLM 直接预测 MovieLens 标准化标签——尚未被覆盖。

 **不同意的重要工作**: TrUMAn (2021) 报告视频-only trope 分类仅 12% 准确率，暗示视频理解极难。本文认为该结论受限于 2021 年的模型能力（BERT+视频编码器），现代 MLLM（2024+）应显著超越。

 ### A-Methodology (2000-2500 words)

 **方法被挑战时的防御**:
 - 挑战: "为什么用 MLLM 而不是专门训练的小模型？" → 回应: MLLM 无需标注训练数据即可做零样本标注，小模型需要大规模标注数据，这正是我们要解决的问题（鸡生蛋问题）
 - 挑战: "genome-scores 是 SVD++ 从用户行为训练的，MLLM 预测的可能是用户偏好而非内容特征" → 回应: 这正是本文要量化的——我们分别评估"内容驱动型标签"（如 "dark atmosphere"）和"偏好驱动型标签"（如 "overrated"），预期 MLLM 在前者上表现更好

 **更简单的方法**: 仅用剧本+TMDb 类型元数据做标签分类（不需要 MLLM），用传统多标签分类器。本文选择 MLLM 因为: (1) 剧本文本无法捕捉视觉/音频特征，(2) MLLM 的零样本能力避免了对标注训练数据的需求。

 **方法最大局限**: 1,128 个 genome tag 数量大，MLLM 无法一次全部评分。分批策略引入批次间不一致性。

 **方法描述结构**:

 3.1 数据准备
 - MovieNet → 多模态块提取（关键帧 + 剧本场景 + 台词/字幕，时间戳对齐）
 - MovieLens genome-scores.csv → ground truth 关联度（1,128 tag × N 电影）
 - MovieNet-MovieLens 对齐：通过电影标题 + 年份匹配

 3.2 MLLM 标注流程
 - Tag 分组策略：将 1,128 tag 按语义聚类为 ~20 组（如视觉风格、情绪氛围、叙事主题、角色特征、制作要素）
 - Prompt 设计：每批输入一组 tag 定义 + 多模态块，要求 MLLM 输出每 tag 的关联度 (0-1)
 - 多次调用取平均（3 次）稳定输出
 - Few-shot 示例：提供 2-3 部已知标签的电影作为示例

 3.3 评估协议
 - 主实验: RMSE / MAE / Precision@10 / NDCG@10 vs genome-scores
 - 消融: video-only / script-only / dialogue-only / 两两组合 / 三模态
 - 时序分割: 2022 年前上映 → 训练参考, 2022 年后 → 测试
 - 冷启动: 无用户标签的电影子集单独评估
 - 一致性: 同一电影 5 次调用的标准差 / ICC

 3.4 对比基线
 - 基线 1: 随机评分
 - 基线 2: 全局均值（每 tag 的全局平均分）
 - 基线 3: genre-only（仅用 genre 预测，如所有 Action 电影给 "action-packed" 高分）
 - 基线 4: 剧本文本 + 传统多标签分类器（如 BERT fine-tune）

 ### A-Results (1500-2000 words)

 **最重要发现（一句话）**: MLLM 在内容驱动型标签（如 "dark", "atmospheric", "visually striking"）上显著优于偏好驱动型标签（如 "overrated", "feel-good"），且三模态组合优于任何单模态。

 **预期结果结构**:
 1. 主实验: MLLM vs 基线的 RMSE/MAE/Precision@K/NDCG@K
 2. 消融: 模态贡献热力图（模态 × 标签类别）
 3. 时序分割: 2022 前后性能对比（推理 vs 回忆的分离）
 4. 冷启动: 无用户标签电影上的标注质量
 5. 一致性: 多次调用的稳定性分析
 6. 案例分析: 2-3 部典型电影的 MLLM 标注 vs ground truth 对比

 **意外结果预案**: 如果 MLLM 在时序分割后性能显著下降（暗示记忆而非推理），这本身是一个有价值的发现——将重写论文叙事为"MLLM 标注的局限性分析"。

 ### A-Discussion (800-1000 words)

 **与文献对话**:
 - vs TrUMAn (2021): 现代 MLLM 是否显著超越 BERT+视频编码器的 12%？差距有多大？
 - vs VLM4Rec (2026): VLM 生成自然语言描述 vs 直接预测标签评分，哪种更适合推荐？
 - vs ViLLA-MMBench (2025): LLM 生成摘要 vs MLLM 直接看内容，哪种更有效？

 **最想让读者记住的**: 视频模态对"视觉风格类"标签的边际贡献，证明了"看懂电影"对标签预测的独特价值——这是纯文本方法无法替代的。

 ### A-Conclusion (400-500 words)

 **如果只能留一段话**: MLLM 能够从电影多模态内容预测 MovieLens genome tag 关联度，尤其在内容驱动型标签上表现优秀，为推荐系统冷启动提供了可行的自动化标注路径。但 MLLM 存在数据记忆风险和偏好驱动型标签上的局限，需在时序分割上谨慎评估。

 ## A-Step 2.5: 贡献锐化

 > **[INSIGHT: contribution_claim]**
 > 本文的核心贡献是方法层面的：首次提出将 MLLM 的多模态理解能力与 MovieLens genome tag 评分体系对齐的方法框架，并通过系统的模态消融和时序分割评估，量化了"真正看懂电影"对标签预测的增量价值。

 ## A-Step 3: 论证压力测试

 **核心论证链**:
 1. MovieLens genome tag 当前依赖用户行为训练 SVD++，冷启动场景下无法生成 → 引 Harper & Konstan (2015)
 2. MLLM 2024 年后具备多模态理解能力，可从内容直接推理标签 → Technology Cycle 探针
 3. 三模态组合 > 任何单模态 → 消融实验验证（待数据确认）
 4. 时序分割后性能不显著下降 → 推理而非回忆（待数据确认）
 5. 冷启动场景下 MLLM 标注质量显著优于随机/均值基线 → 冷启动实验（待数据确认）

 **最弱点**: 论证链第 4 步——如果时序分割后性能确实下降，核心论点从"MLLM 能推理标签"变为"MLLM 有记忆风险"，论文叙事需要调整。

 **论证反转测试**: 如果 MLLM 标注质量不如简单的 genre-only 基线（genre → tag 映射表），那么核心论点不成立。防御: genome tag 远比 genre 细粒度（1,128 vs 20），genre-only 基线无法区分 "dark" vs "noir" vs "bleak"。

 ## A-关键文献定位表

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

 ## A-时间线

 | 阶段 | 内容 | 时长 | 产出 |
 |------|------|------|------|
 | M1-M2 | 数据准备: MovieNet-MovieLens 对齐 + tag 分组 | 2 月 | 对齐数据集 + tag 分组方案 |
 | M2-M4 | MLLM 实验主跑 + 消融 + 时序分割 | 2 月 | 实验结果表 + 热力图 |
 | M4-M5 | 冷启动实验 + 一致性分析 + 案例分析 | 1 月 | 补充实验结果 |
 | M5-M6 | 论文写作: Intro + Related Work + Method | 1 月 | 前半部分初稿 |
 | M6-M7 | 论文写作: Results + Discussion + Conclusion + Abstract | 1 月 | 完整初稿 |
 | M7 | 内部评审 + 修改 | 0.5 月 | 终稿 |
 | **总计** | | **6.5 月** | |

 ## A-目标场所

 RecSys 2027 (ACM Conference on Recommender Systems) 或 CIKM 2027
 论文类型: Novel Method（IMRaD 结构）
 引用格式: ACM (会议标准)
 预估字数: 6000-8000 words (正文)

 ---

---

 # 三方向横向对比

 | 维度 | A: GenomeTag-MLLM | B: MovieTagGen | C: ModalTag-Decomp |
 |------|-------------------|---------------|-------------------|
 | 论文类型 | Novel Method | Novel Method | 分析型 |
 | 核心贡献 | 方法（MLLM→genome 对齐） | 方法（两阶段+审计） | 分析（模态贡献分解） |
 | 论证风险 | 时序分割后性能下降 | 两阶段误差累积 | 发现太直觉性 |
 | Ground Truth | genome-scores.csv (干净) | tags.csv + genome (复杂) | genome-scores.csv (干净) |
 | 评估复杂度 | 中 | 高（含人工评估） | 低（消融即评估） |
 | 预估周期 | 6.5 月 | 8 月 | 5.5 月 |
 | 目标场所 | RecSys / CIKM | WWW / SIGIR | MM / RecSys |
 | 字数 | 6-8K | 7-9K | 5-7K |
 | 独立性 | 可独立 | 可独立（但最好在 A 之后） | 可独立（但最好在 A 之后） |
 | 组合建议 | 先做 A | A 完成后做 B 扩展 | A 或 C 先做均可 |

 **推荐执行顺序**: C → A → B
 - C 最快（5.5 月），先做建立模态理解基础
 - A 基于 C 的理解设计最优模态组合（6.5 月）
 - B 在 A 的基础上扩展为两阶段（8 月，但 A 的实验可复用）
 - 总计 ~20 月发 3 篇，第二年起每年 1-2 篇产出

 ---

 *以上三个研究计划基于 academic-paper plan 模式框架生成，已包含核心论点、章节计划、论证压力测试、文献定位表和时间线。可直接用于启动论文写作（切换到 full 模式）或提交导师审阅。*
