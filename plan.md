作业实现计划（基于 Elliot 框架）

目标
- 基于 MovieLens 标签数据验证“基于标签的推荐算法”，并对标签清洗与可解释性进行分析。
- 覆盖多类方法：基于内容/近邻（低复杂度）、基于因子分解（FM 作为可行替代）、基于图（RP3beta 基线）、基于 LDA 主题（将主题当作侧信息特征）。
- 使用 Elliot 的统一实验管线：数据加载→划分→训练→评测→结果与统计。

仓库与用法（简述）
- 入口：`elliot/run.py:1` 的 `run_experiment(config_path)`；示例启动脚本 `start_experiments.py:1`。
- 配置核心结构（YAML）：
  - `experiment.dataset`、`data_config.strategy: dataset`、`data_config.dataset_path`（制备成 TSV：userId itemId rating timestamp）。
  - `data_config.side_information`：可挂载侧信息加载器（如 `ItemAttributes`），供依赖标签/特征的模型使用。
  - `splitting`：随机/时间/交叉验证等；`evaluation`：指标、cutoffs；`models`：模型与超参网格。
- 现成的与本作业相关模块：
  - 侧信息加载器：`ItemAttributes`（elliot/dataset/modular_loaders/generic/item_attributes.py:1）读取 `itemId \t featureId...`。
  - 基于标签/特征的模型：
    - `VSM`（TF-IDF/Binary 用户与物品标签画像）`elliot/recommender/content_based/VSM/vector_space_model.py:1`。
    - `AttributeItemKNN`、`AttributeUserKNN`（特征空间相似度近邻）。
    - `FM`（Factorization Machines，可融合用户/物品/特征）`elliot/recommender/latent_factor_models/FM/factorization_machine.py:1`。
  - 基于图基线：`RP3beta`（点击/交互图随机游走变体）`elliot/recommender/graph_based/RP3beta/rp3beta.py:1`。

与作业要求的差距与需要补充
- 数据：必须使用含标签的 MovieLens（如 ml-20m 或 ml-latest-small），仓库示例默认抓取 ml-1m（无 tags）。需新增数据制备脚本：
  - 从 GroupLens 下载 ratings.csv、tags.csv，生成：
    - `data/<dataset>/dataset.tsv`（评分数据，制作为 TSV）。
    - `data/<dataset>/item_features.tsv`（每行 `itemId \t tagId1 \t tagId2 ...`）。
    - `data/<dataset>/tags_map.tsv`（`tagId \t tag_text`，用于解释与分析）。
- LDA：内置无 LDA 主题模型；可离线用 sklearn/gensim 训练 LDA，将每个物品 Top-K 主题当作“特征”写为 `item_topics.tsv` 再以 `ItemAttributes` 挂载。
- “张量分解”严格的三元（user–item–tag）分解（如 PITF/RTF）库内未提供。如需严格 3 阶张量，可通过 `external_models_path` 接口引入自定义模型（可选扩展）。
- 若希望加载“带权重”的特征（如 LDA 主题概率），当前 `ItemAttributes` 仅支持离散特征 ID。两个可行折中：
  - 使用 `VSM` 的 TF-IDF 重新加权（不读入外部权重）；
  - 写一个简单“WeightedItemAttributes”加载器（可选扩展，非必须）。

实验设计
- 数据集与划分
  - 开发/验证先用 `ml-latest-small`（小、快），最终在 `ml-20m` 复现实验。
  - 划分：`temporal_hold_out`（时间保留）或 `random_subsampling`，测试比 0.2。
  - 二值化：可选（评分→隐式1），根据指标选择；默认保留显式评分但多数模型按隐式评测。
- 标签清洗（tags.csv → item_features.tsv）
  - 统一小写、去标点/数字、去左右空白、合并连续空白。
  - 停用词过滤、去除过短词（长度<2）和低频词（全局频次 < min_count，默认 5/10，做消融）。
  - 每物品去重；可限制每物品最多 Top-N 标签（按局部频次/TF-IDF，N=20/50 做对比）。
  - 产出 tagId 映射文件（便于可解释与分析报告）。
- LDA 主题（可选）
  - 基于清洗后的标签为每部电影构建“文档”，训练 K 主题（K∈{20,50,100} 网格），保留每物品 Top-3/Top-5 主题 ID 作为特征。
- 算法与配置
  - 标签/特征驱动：
    - VSM：`similarity ∈ {cosine, correlation}`；`user_profile ∈ {tfidf, binary}`；`item_profile ∈ {tfidf, binary}`。
    - AttributeItemKNN / AttributeUserKNN：`neighbors ∈ {15, 30, 60, 100}`；`similarity ∈ {cosine, correlation}`；`profile ∈ {binary, tfidf}`。
    - FM：`factors ∈ {16, 32, 64}`；`lr ∈ {1e-3, 5e-4}`；`reg ∈ {1e-2, 1e-1}`；`loader: ItemAttributes`。
  - 图基线：RP3beta：`neighborhood ∈ {50, 100, 200}`；`alpha ∈ {0.8, 1.0}`；`beta ∈ {0.3, 0.6}`。
  - LDA 特征替换：将 `attribute_file` 切换为 `item_topics.tsv`，在 VSM/FM 上复现实验。
- 评测设置
  - 指标：`nDCG@10, Recall@10, MAP@10, MRR@10`；可选多 cutoffs（@20, @50）。
  - 多样性/覆盖率（可选）：`SRecall` 需要 `feature_map`（即标签侧信息）以统计子主题覆盖。
  - 统计检验：Elliot 内置配对 t 检验/Wilcoxon（非折叠划分时可用）。
- 结果与可解释
  - 为每个模型保存推荐结果（`save_recs: True`）。
  - 用 `tags_map.tsv` 将物品特征 ID 反查为文本，展示：用户画像 Top 标签、物品 Top 标签、推荐条目的“标签交集/相似度”示例。

落地实现（代码与文件）
- 数据制备脚本：`scripts/prepare_ml_tags.py:1`
  - 输入：选择 `ml-latest-small` 或 `ml-20m`；下载/解压 → 生成 `dataset.tsv`、`item_features.tsv`、`tags_map.tsv`（可选未来扩展：`item_topics.tsv`）。
  - 参数：`--dataset {small,20m}`，`--min-count`，`--max-tags-per-item`。
- 实验配置：`config_files/tag_experiments.yml:1`
  - `data_config.dataset_path: ../data/<dataset>/dataset.tsv`
  - `side_information: [{ dataloader: ItemAttributes, attribute_file: ../data/<dataset>/item_features.tsv }]`
  - `models:` 包含 VSM、AttributeItemKNN、AttributeUserKNN、FM、RP3beta 的超参网格。
- 运行方式
  - `python start_experiments.py --config tag_experiments`（参见 `start_experiments.py:1`）。
  - 先生成数据（示例，small 版本）：
    - `python scripts/prepare_ml_tags.py --dataset small --min-count 3 --max-tags-per-item 50`
    - 生成路径：`data/movielens_small/{dataset.tsv,item_features.tsv,tags_map.tsv}`

里程碑与进度追踪
- [x] 数据脚本：ratings/tags 解析与清洗（small → 20m）。文件：`scripts/prepare_ml_tags.py:1`
- [x] 配置 `tag_experiments.yml`（含模型与超参网格）。文件：`config_files/tag_experiments.yml:1`
- [ ] 生成 `dataset.tsv`、`item_features.tsv`、`tags_map.tsv`。
- [ ] 跑通 small 数据集端到端，产出指标与推荐清单。
- [ ] 标签清洗消融（min_count、max_tags_per_item、TF-IDF vs Binary）。
- [ ] 图基线（RP3beta）对比。
- [ ] FM 对比（标签侧信息）。
- [ ] LDA 主题版特征与对比（可选）。
- [ ] 结果分析与可解释展示（用户/物品 Top 标签 + 推荐解释）。
- [ ] 撰写报告与结论。

备注
- 若需要严格的 user–item–tag 三元张量分解（如 PITF），建议以 `external_models_path` 扩展外部模型，或在当前作业中以 FM 作为“基于因子分解”的可行替代方案。
- Elliot 对侧信息的读取以 `ItemAttributes` 为主，若后续需要“带权重特征”，可在 `elliot/dataset/modular_loaders/generic` 下新增轻量加载器（可选）。
