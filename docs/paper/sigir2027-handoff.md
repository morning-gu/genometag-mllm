# SIGIR 2027 论文交接文档（Handoff）

> **文档目的**：本文档是唯一的交接载体。接手的 AI 工具（或人类协作者）在**不接触本会话**的前提下，凭此文档 + 文档中列出的仓库/服务器文件，能够撰写并打磨出可投稿的 SIGIR 2027 英文全文。
> **写作状态**：实验已全部完成并冻结（2026-10-28），无需任何新实验。待办只有：写作、内审、投稿。
> **数字铁律**：本文档中所有数字均为最终实验值，与服务器结果文件一致。写作时**禁止编造或四舍五入更改任何数字**；如需本文档未覆盖的数字，到 §12 文件地图所列文件中查找，找不到就留 TODO 标记，不得虚构。

---

## 1. 一页纸摘要

**研究问题**：多模态大语言模型（MLLM）能否从电影内容（关键帧+剧本+台词）预测 MovieLens Genome 的 1,128 维标签关联度评分？——更重要的：**它到底在使用内容，还是在回忆训练数据里见过的电影？**

**核心发现（诚实负面 + 机制修正）**：MLLM 的标签能力主要由**记忆先验**（title-only 盲测 NDCG：老片宇宙 0.76 / 新片 0.66，全部口径）承载；内容通路**真实存在但被再分配**（错配探针 corr_B 0.61 ≫ corr_A 0.27-0.35 证明内容被因果使用；剂量-反应显示弱先验电影内容增益 +0.09~+0.14、强先验电影内容拖累 −0.10~−0.14），在著名老片宇宙上正负抵消为"净贡献≈0"，在先验衰减的 ≥2015 新片上净方向转正。偏好型标签（"overrated" 类）逐片显著偏输（p=0.0073）。校准（RMSE）全线显著劣于统计基线。

**论文类型**：评估 + 审计方法论（非"新方法赢了"）。三贡献：①内容-记忆分解探针套件（可复用方法论）②四层证据的系统性评估（1,035 部对齐语料、6 臂消融、150 对错配、70 部新片臂）③对"加多模态内容必然更好"这一流行假设的量化修正与边界刻画。

**一句话叙事**：*Not ignored, not redundant — reallocated: what MLLMs actually do with movie content when tagging items.*

---

## 2. 目标场所与硬约束

| 项 | 值 | 备注 |
|---|---|---|
| 首投 | **SIGIR 2027** 全文 | 截稿约 **2027-01 中旬**（⚠️ 动手前核实当年 CFP 的准确日期与页数上限；近年为正文 9 页 + 参考文献不限，ACM sigconf 双栏） |
| 级联备胎 | CIKM 2027 | 截稿约 2027-05 底；SIGIR 4 月出通知后可转投（吸收审稿意见、贡献排序调回评估优先） |
| 匿名 | **双盲**：正文与致谢不得出现作者名、单位、可识别的仓库链接 | 引用自己的前置工作用第三人称 |
| 语言 | 英文 | ACM sigconf 模板；LaTeX 直接写 |
| 字数体量 | 正文 6,000-8,000 words（9 页双栏上限内） | 图表占位需预留约 2 页 |
| 时间线 | 2026-11 初稿骨架 → 2026-12 全文 → 2027-01 上旬内审打磨 → 1 月中投稿 | 今天是 2026-10-28，剩约 10-11 周 |

**SIGIR 叙事适配**（当初选级联的原因）：SIGIR 检索内核，纯"标注保真度"评估有 relevance 风险——所以**贡献排序必须是方法论优先**（见 §3）。转投 CIKM 时只改引言与贡献列表顺序（评估发现升回第一位），正文主体不动。

---

## 3. 叙事设计与贡献列表

**标题候选**（择一或改写，英文）：
1. *Not Ignored, Not Redundant — Reallocated: Auditing What Multimodal LLMs Actually Use When Tagging Movies*
2. *Memory Prior, Real Content: A Probe Suite for Decomposing Multimodal LLM Item Tagging*
3. *Auditing Content Use in MLLM Item Annotation: Crossover Probes and Dose-Response Evidence from 1,128 Movie Tags*

**贡献列表草案**（Intro 结尾三句式）：
1. **A probe suite for decomposing content vs. memory in MLLM item annotation**：title-only blind arm（零内容块消融）+ mismatched-content crossover（A 标题 × B 内容，150 对，同/跨类型对照）+ prior-conditional dose-response regression（含对 Δ~title 部分-整体相关陷阱与斜率衰减偏误的显式规避——§9 红线）。探针套件模型无关，可直接复用于任何 item 标注/元数据生成管线。
2. **A systematic, leakage-controlled evaluation** on MovieLens Genome × MovieNet（1,035 部对齐；电影级留出协议；6 模态臂 × 4 统计基线；3 轮/任务；全部代码与清单公开可复现）。
3. **Empirical corrections to a prevailing assumption**：内容净贡献≈0 的机制是**再分配抵消**（非忽略、非冗余）；preference 型标签显著受损；校准是独立于排序能力的短板。

**叙事红线**：全文不得暗示"我们提出了更好的标注方法"——方法是审计，结论是边界。语气参照诚实负面结果文献（负结果 ≠ 失败，是修正社区先验）。

---

## 4. 证据链与全部关键数字

> 四个实验块按论证顺序排列。每块给出写作必需的全部数字。

### 4.1 主实验（协议 v3 六臂，88 部测试集）

- **协议**：评估宇宙 = 三模态全有 **440 部**；测试集 = 年代分层随机 **88 部**（20%，seed 42，清单 `data/processed/eval_test.txt` 入 git）；拟合池 = 补集 **947 部**（few-shot 与统计基线仅用拟合池，防泄露断言内建）。任务总量 52,213 done / 59 failed（null 1.4%），298M input + 118M output tokens。
- **配置**：qwen3.7-plus（DashScope，思考模式开，~2,400 out tok/任务），temperature 0.7，每任务 3 轮取平均，12 关键帧 + 剧本 4,000 tok + 台词 3,000 tok，33 个语义组（覆盖全部 1,128 tags），few-shot 2 例 = Goodfellas (1990) + The Terminator (1984)。
- **主表数字**（06 官方评估输出 `results/v3-*/evaluation/metrics.csv`；per-movie NDCG@10，truth 阈值 0.5；RMSE 为全部口径）：

| 臂 | NDCG 全部 | NDCG content | NDCG pref | RMSE |
|---|---|---|---|---|
| genre 基线 | **0.789** | 0.769 | 0.644 | 0.135 |
| title-only | 0.764 | **0.781** | 0.614 | 0.206 |
| video+script | 0.763 | 0.743 | 0.574 | 0.233 |
| dialogue-only | 0.756 | 0.758 | 0.594 | 0.219 |
| full（三模态） | 0.754 | 0.754 | 0.584 | 0.230 |
| video-only | 0.747 | 0.767 | 0.592 | 0.237 |
| script-only | 0.741 | 0.744 | 0.578 | 0.220 |
| mean 基线 | 0.697 | 0.662 | 0.579 | 0.150 |
| random 基线 | 0.064 | 0.050 | 0.141 | 0.497 |

- **主表要点**：①content 口径上 title-only（0.781）**超过 genre 基线**（0.769）——记忆先验在内容型标签上已是天花板级；②preference 口径 MLLM 全员大跌（full 0.584 vs genre 0.644）；③RMSE 与排序能力脱钩。每臂任务数 **8,712**（88 部 × 33 组 × 3 轮）。
- **显著性层**（逐片配对 bootstrap 10k；与 06 官方表可能有 ≤0.002 的聚合差——论文表格以 06 官方为准，p 值以逐片配对为准。重算路径：`genometag/eval/per_movie.py` + `genometag/eval/metrics.py` 的 `paired_bootstrap`）：full−title-only 全部 **−0.013** (p=0.475 ns) / content **−0.028** (p=0.062 边缘) / preference **−0.032** (**p=0.017 显著负**)；full vs genre 全部 −0.037 (p=0.079)、preference −0.062 (p=0.031)；title-only vs genre on content +0.012 (p=0.577 ns)；title-only vs mean 全部 +0.068 (p=0.006)；RMSE full 0.230 vs genre 0.135 (**p<0.0001**)。

### 4.2 探针 1 · title-only 盲测

即上表 title-only 臂。要点：prompt 仅标题+年份、零内容块；与消融臂同测试集同协议并列，故差值是受控消融而非跨实验比较。title-only 0.765/0.782 本身即"记忆先验强度"的量化。

### 4.3 探针 2 · 错配内容交叉（mismatch probe）

- **设计**：150 对（75 同类型 + 75 跨类型），prompt = 电影 A 标题 + 电影 B 的内容块，33 组全跑，4,950 任务（null 4,027 为标签级缺失，宽松解析设计使然，非失败）。
- **度量**：逐对 Pearson(pred, truth_A) 与 Pearson(pred, truth_B)；shift = corr_B − corr_A。
- **结果**：

| 条件 | n | corr_A | corr_B | shift |
|---|---|---|---|---|
| 跨类型 | 75 | 0.273 | **0.611** | **+0.338** |
| 同类型 | 75 | 0.350 | **0.621** | **+0.270** |

- **解读**：预测强烈跟随错配内容 B 而非标题 A 的真值 → 内容被**因果使用**且证据压过标题先验；同类型 corr_A 更高佐证先验与 B 真值因 genre 相关部分对齐，稀释可归因偏移。这排除了"模型忽略内容"的解释——v3 的零净贡献必是下述两条之一：冗余或再分配（由 4.4 裁决）。
- **corr 计算细节**（方法节需写明；实现在 `genometag/probes/mismatch.py`）：逐对计算，预测标签集与 A、B 各自的 genome 真值取交集，NaN 掩码后任一侧 std=0 计 0，否则 Pearson。

### 4.4 ≥2015 新片臂（先验衰减人群，70 部）

- **设计**：年份 ≥2015 且帧+字幕可得 70 部（年份计数 95，模态过滤后 70；与 88 部测试集交集强制为 0）；双臂 full(video|dialogue) vs title-only，13,860 任务（full 臂 6,916 done/14 failed；title 臂 6,930/0）。注意 2 模态口径，与老片 3 模态臂**不可直接混比**，仅在人群内自比。
- **臂级**（06 官方，全部口径）：full **0.679** / title-only **0.663** / genre 0.664 / mean 0.540 / random 0.047；content 口径 full 0.681 / title 0.691；preference 口径 full 0.389 / title 0.433（**子集口径 full 反而落后**——全集 top-k 竞争效应，与深化分析的非可加性呼应）；RMSE full 0.237 vs genre 0.126。
- **先验门**（同口径对比，06 官方）：全部 0.663 vs 0.764（**−0.102**）；content 0.691 vs 0.781（−0.090）；preference 0.433 vs 0.614（**−0.181**，衰减最大——"overrated"类文化共识标签的记忆需要时间沉淀，Discussion 素材）。
- **主检验**（full~title OLS 斜率 vs 冗余零假设 b=1；movie 级配对 bootstrap 10k）：**b = 0.712 [0.542, 0.863]，p(b=1) = 0.0002** → 拒绝冗余。
- **headline**：mean Δ = **+0.022** [−0.013, +0.055]，p(Δ≤0)=0.107 —— 方向转正、n=70 功效不足。**写作口径**：不声称显著，明确功效限制；显著的是斜率与分箱形态。
- **分箱**（title 四分位 → mean Δ）：Q1 **+0.094**(n=18) / Q2 +0.036(n=17) / Q3 +0.061(n=17) / Q4 **−0.099**(n=18)。
- **老片对照**（v3 数据重算，n=88）：mean Δ −0.013（与主表一致，交叉验证通过）；斜率 b=0.468 [0.292, 0.665]；分箱 Q1 **+0.139** / Q2 −0.040 / Q3 −0.041 / Q4 **−0.112**。
- **合并解读**：两人群同形剂量-反应；老片宇宙正负抵消成零均值，新片宇宙（先验弱 0.119）净方向转正。

### 4.5 深化画像（win-rate / 类型 / 案例）

- **Win-rate（逐片 full>title-only，精确二项 + Wilson 95% CI）**：

| 人群·类型 | W/L/T | rate | p |
|---|---|---|---|
| newfilm·all | 42/27/1 | **0.609** | 0.091 |
| newfilm·content | 31/37/2 | 0.456 | 0.545 |
| newfilm·preference | 24/36/10 | 0.400 | 0.155 |
| old·all | 36/49/3 | 0.424 | 0.193 |
| old·content | 35/47/6 | 0.427 | 0.224 |
| old·preference | 28/53/7 | **0.346** | **0.0073** |

- **非可加性现象**（Discussion 素材）：newfilm delta_all=+0.022 但 content（−0.010）与 preference（−0.044）子集均负——增益来自全集 top-k 竞争（相关标签挤入 top-10），非子集内排序改进；三种口径不可加和。
- **案例（天花板效应，论文用 2-3 个即可）**：受益 = 中低先验（Mad Max: Fury Road prior 0.70→full 1.00, Δ+0.30；The Big Short 0.79→1.00；Surrogates 0.26→0.78, **+0.52**；Margin Call 0.32→0.68, +0.36）；受损 = 先验饱和爆款/经典（Captain America: Civil War prior **1.00**→0.71, −0.29；Die Hard **1.00**→0.48, **−0.52**；Vertigo 1.00→0.80）。
- **类型一致性**：跨人群均增益 = Romance（老 +0.063/新 +0.041）、Comedy（+0.013/+0.038）；跨人群均受损 = **Thriller（−0.042/−0.011）、Fantasy、Mystery**；仅老片受损 = Crime（−0.055；新片 Crime **+0.050 居增益榜首**）、Action（−0.029；新片 +0.003 ≈ 0）；Horror 反转（新 0.80 win n=5 vs 老 0.23 win n=13，小样本存疑）。
- **年代梯度**（老片）：2010s +0.011 → 2000s +0.003 → 1990s −0.027 → 1980s −0.084（n=9；1960s +0.178 为 n=2 反例）。

---

## 5. 图表清单（论文成品 ↔ 数据源）

| # | 形态 | 内容 | 数据源 |
|---|---|---|---|
| T1 | 主表 | 六臂 × 4 基线：NDCG/P@10/RMSE/MAE（全部/content/preference 三视图）+ 配对显著性 | 官方行 `results/v3-*/evaluation/metrics.csv`（已含全部/tag:content/tag:preference 与 4 基线）；显著性重算 = `genometag/eval/per_movie.py` + `metrics.py:paired_bootstrap`（勿依赖服务器 `/tmp/v3_analysis.py`——临时文件，逻辑已固化进仓库） |
| T2 | 探针表 | 错配探针 corr_A/corr_B/shift × 同/跨类型 | `results/v3-mismatch/mismatch_summary.json` |
| T3 | 剂量-反应表 | 两人群：斜率 + CI + p(b=1)；mean Δ + CI；四分位分箱 | `results/newfilm-full/dose_response/dose_summary.json` |
| F1 | 散点+拟合线 | 逐片 title-only NDCG vs Δ，双人群（蓝 newfilm/橙 old） | `results/newfilm-full/dose_response/dose_response.png`（已产出，可重绘为出版级） |
| F2 | ECDF | 双人群逐片 Δ 分布 | `results/newfilm-full/deep_analysis/delta_ecdf.png` |
| F3 | win-rate 点+CI | 6 行（2 人群 × 3 类型）Wilson CI，`*` 标 p<0.05 | `results/newfilm-full/deep_analysis/winrate_by_type.png` |
| F4 | 发散条形 | newfilm genre mean Δ（n≥3） | `results/newfilm-full/deep_analysis/genre_delta_newfilm.png` |
| T4 | 案例 | 受益/受损各 2-3 部（片名+prior+full+Δ） | `results/newfilm-full/deep_analysis/movers_*.csv` |
| T5 | 年代/类型附表 | 老片年代梯度、两人群 genre 聚合 | `results/newfilm-full/deep_analysis/deep_summary.json`（by_decade 字段）、`delta_by_genre_{newfilm,old}.csv` |

已产出的诊断图是 matplotlib 草稿级，出版重绘时保持配色语义（蓝=新片/实验人群，橙=老片/对照；发散条形蓝正橙负）。

---

## 6. 章节规划（SIGIR 口径，方法论优先）

| 节 | 字数 | 内容要点 |
|---|---|---|
| Abstract | ~200 | 问题 → 探针套件 → 四层证据三个数字（prior 0.76 vs 0.66；shift +0.34；b=0.71, p=0.0002）→ 修正性结论 |
| 1 Introduction | 850 | 假设困境（genome 标签依赖用户行为，冷启动不可得）→ "上 MLLM 看内容"的流行做法 → 审计问题：它真在用内容吗 → 贡献三句（§3） |
| 2 Related Work | 1,000 | 三线：genome 标签体系与局限 / 电影多模态理解（mise-en-scène→MovieCLIP→VLM4Rec）/ MLLM×推荐元数据（ViLLA-MMBench、TrUMAn、TiM）。对话点：TrUMAn 报告视频-only trope 分类仅 12% 准确率，现代 MLLM 是否颠覆？答案：颠覆方式出乎预料（内容被使用，但先验仍主导） |
| 3 Task & Data | 850 | genome 评分体系（1,128 tags，relevance 长尾 mean 0.111、≥0.5 仅 3.6%）× MovieNet（1,035 部对齐；剧本 480/字幕 816 覆盖偏差如实报告）→ content/preference 类型分解的动机 |
| 4 Probe Suite & Protocol | 1,550 | 3.1 标注管线（分组+定义+多轮平均）3.2 协议 v3（电影级留出+防泄露）3.3 探针 1/2/3（重点写：为什么 Δ~title 直接回归是错的——部分-整体陷阱；为什么主检验是 slope-vs-1；衰减偏误声明） |
| 5 Results | 1,700 | 5.1 主表与消融（净贡献≈0）→ 5.2 错配探针（内容被因果使用）→ 5.3 新片臂剂量-反应（再分配定裁）→ 5.4 画像（win-rate/案例/类型） |
| 6 Discussion | 850 | 对 TrUMAn 的代际修正；对"加内容特征"实践的三条可操作含义（先审计先验、弱先验段才值得上内容、preference 标签别用 MLLM）；校准短板与 RMSE |
| 7 Limitations & Conclusion | 650 | §9 全部 limitation + 结论 |

---

## 7. 方法事实卡（写作时照抄，勿凭记忆改写）

- 模型：**qwen3.7-plus**（DashScope OpenAI 兼容端点），思考模式默认开，temperature 0.7；每任务 3 轮取均值。
- 输入预算：12 关键帧 + 剧本 4,000 token + 台词 3,000 token；输出 JSON `{tag: score∈[0,1]}`；33 个语义组（每组 ≤60 tags，组内带一句话定义；分组与 content/preference 类型标注由 qwen3.8-max 生成）。
- few-shot：2 例，仅从拟合池选（top-2 相似度 = Goodfellas 1990、The Terminator 1984），各臂一致。
- 解析：宽松解析（合法条目全收、缺失记 null）——null 率 1.1-1.4%（主实验）、标签级 ~0.8/任务（探针）。
- 工程：SHA256 缓存（prompt+model+params+run）、jsonl 断点续跑、8 线程、看门狗自动重启零损失。
- 指标：NDCG@10 / Precision@10 / RMSE / MAE；relevance 阈值 0.5；per-movie 计算 + 配对 bootstrap（10k，seed 42）；win-rate 用精确二项检验 + Wilson 95% CI。
- 统计口径细节（防审稿人挑刺，也防写作时口误）：①逐片 NDCG 先对 3 轮取均值、与 truth 内连接（truth 缺失标签剔除）②与官方评估路径**同序**（tag_id 升序喂 ndcg_at_k，并列分数下才与 06 逐位一致）③斜率检验的 bootstrap 剔除零方差自抽样产生的 NaN/±inf。

---

## 8. Limitations 清单（Limitations 节直接展开）

1. **genome scores 由用户行为（SVD++）训练**——MLLM 预测的是"用户会怎么标"而非"电影内容是什么"；content/preference 分解部分回应但未消除。
2. **合法世界知识 vs MovieLens 记忆不可完全分离**——title-only 先验包含两者；探针量化的是"内容增量"，不是"记忆来源"。
3. **单一模型家族**（qwen3.7-plus）——结论的模型外推性未测。
4. **分组评分引入组间条件差异**——1,128 tags 无法单次评分，33 组批处理以组内定义+多轮平均缓解。
5. **剂量-反应斜率含测量误差衰减偏误**（title NDCG 逐片噪声使斜率向下偏）——两人群同向偏，跨人群**不比较斜率大小**，结论锚定在分箱形态与方向。
6. **新片臂 2 模态口径**（剧本缺失）——与老片 3 模态臂不可直接混比。
7. **n=70 功效限制**——headline Δ 未显著（p=0.107）。
8. **MovieNet 内容止于 2018、剧本覆盖偏向 90s-00s**——部署现实的内容覆盖偏差。
9. **英文电影语料为主**——多语言泛化未测。

---

## 9. 写作红线（违反即翻车）

1. **Δ~title 直接回归并报告负斜率 = 方法论自杀**。部分-整体相关：若 full⊥title，期望斜率恰为 −1。主检验必须是 full~title 斜率 vs 1（冗余零假设）；Δ~title 的 Spearman 只能作描述并明示警告。
2. **斜率跨人群比较被禁止**（衰减偏误人群异质）；只比方向与分箱形态。
3. **headline Δ 不得写成"显著"**（p=0.107）；显著的是斜率 p=0.0002 与 old·preference win-rate p=0.0073。
4. **ndcg_all ≠ mean(ndcg_content, ndcg_preference)**（全集 top-k 竞争效应），三种口径分开报告，不得互相推算。
5. **不得编造引用**：§10 文献表中标 ⚠️ 的条目未经验证，引用前必须逐条核实（给出真实标题/作者/年份/venue；对不上就删，宁缺毋假）。
6. **双盲**：正文不出现作者/单位/自家仓库直链；代码与数据可用匿名声明（"code will be released"）表述。
7. **数字铁律**（见文档头）：所有数字以本文档与 §12 文件为准。
8. **术语一致**（首次出现给定义，全文统一）：memory prior / content pathway / reallocation / mismatched-content crossover / dose-response / win-rate / blind arm。

---

## 10. 文献定位表（⚠️ = 未核实，引用前必须验证）

| 角色 | 文献 | 状态 |
|---|---|---|
| MovieLens 官方引用 | Harper & Konstan (2015), "The MovieLens Datasets" | ✅ 可靠 |
| genome 体系 | Vig et al. (2012), "The MovieLens Genome" | ✅ 可靠（建议补引） |
| 视觉特征先驱 | Deldjoo et al. 2017 | ⚠️ 核实具体篇目 |
| 最近邻 | TrUMAn (2021) — 报告视频-only trope 分类 ~12% 准确率 | ⚠️ 核实数字与结论 |
| 最近邻 | TiM (2024) | ⚠️ 核实 |
| 电影场景理解 | MovieCLIP (2022) | ⚠️ 核实 |
| LLM 元数据增强 | ViLLA-MMBench (2025) | ⚠️ 核实 |
| MLLM 描述用于推荐 | VLM4Rec | ⚠️ 核实 |
| 多模态数据集 | MovieNet (2019) | ✅ 可靠 |
| 文本标签预测基线 | Folksonomication (2018) | ⚠️ 核实 |
| 记忆/泄露警示 | ML-1M memorization 警示文 | ⚠️ 核实（该文与本工作动机高度呼应，值得花力气找全） |

**侦察任务**（写 Related Work 前必做）：以 "LLM item metadata"、"multimodal LLM content use audit/probe"、"MLLM memorization recommendation"、"cold-start tag generation" 为关键词补一轮 2025-2026 文献；本文的探针方法论很可能有平行工作， Related Work 必须主动划界（我们的差异：评分式标注 vs 描述生成、crossover 因果设计、先验条件化分析）。

---

## 11. 章节骨架之上的写作指令（给接手 AI 的操作建议）

1. **先写 §4（Probe Suite & Protocol）和 §5（Results）**——事实密度最高、最不受风格影响；Intro/Abstract 最后写（叙事才稳）。
2. 每完成一节做一次"审稿人三问"自查：这个声明有数字支撑吗？数字与 §4-§5 一致吗？对立面（"内容当然有用""这就是常识"）被正面处理了吗？
3. 表格直接从 §5 清单列出的 JSON/CSV 生成，禁止手抄数字进 LaTeX（用脚本导出 LaTeX 表格，杜绝转录错误）。
4. 图：重绘出版级时保持 §5 的配色语义与 CI 语义；坐标轴一律标注口径（per-movie NDCG@10）。
5. 中文思考、英文成文；英文行文忌"我们惊人地发现"式夸张——本文的说服力来自克制的审计语气。
6. 转投预案（若 SIGIR 拒）：CIKM 版改 Intro 贡献排序（评估发现第一、方法论第二）+ 回应审稿意见，其余不动。

---

## 12. 文件地图

**仓库**（GitHub 私库 morning-gu/genometag-mllm，main @ 132f2da；分支 sigir2027-paper 为论文工作分支）：
- 设计与协议：`docs/research-plan.md`（2026-10-19 版，§5 预期结果已过时，以本文档为准）、`docs/v3-results.md`、`docs/tag-groups-zh.md`（33 组 × 1,128 tags 中文清单）
- 探针/分析代码：`genometag/probes/mismatch.py`、`genometag/eval/per_movie.py`、`genometag/eval/deep_analysis.py`、`scripts/{06,07,09,10,11}_*.py`
- 配置：`configs/ablation/*.yaml`（六臂）、`configs/probes/{mismatch,title-only}.yaml`、`configs/newfilm/{full,title}.yaml`

**服务器**（9.15.154.212，项目 `/cache/g00619218/genometag-mllm`）：
- 主实验：`results/v3-{full,title-only,video-only,dialogue,video-script,script-only}/`（annotations.parquet + evaluation/）
- 探针：`results/v3-mismatch/{mismatch_summary.json,mismatch.parquet}`
- 新片臂：`results/newfilm-{full,title}/`（annotations + evaluation）
- 深化：`results/newfilm-full/dose_response/`、`results/newfilm-full/deep_analysis/`
- 访问方式：`SRV_PASS='mt2021@linux@RD' bash C:/Users/admin/AppData/Local/Temp/srv.sh "<远程命令>"`（askpass 助手；仅本机可用）

**项目记忆**：`C:/Users/admin/.claude/projects/C--code-research-topics-movie-tag-research/memory/genometag-mllm-project-state.md`（本会话沉淀的全部项目状态）

---

## 13. 待用户确认项（接手 AI 开工前问一次）

1. 署名与单位（双盲稿不出现，但投稿系统需要）；
2. LaTeX 编译环境（本地 MiKTeX/tectonic 或 Overleaf）；
3. 是否公开代码仓（影响正文"code will be released"表述与匿名化处理）。
