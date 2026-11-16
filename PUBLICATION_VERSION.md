> 说明：官方源码快照 `official_publication_source/` 不随本仓库分发；正文提到的其它 artifacts 审计文件需在获取官方快照后运行审计脚本生成。如需运行等价性检查，请从作者仓库获取提交 `f00df860`。

# 发表前版本与提供版本的差异

本文保存源码版本和历史适配差异。当前主实现 `paper_algorithm_v4` 已进一步修正论文与源码的算法差异，详见 [PAPER_ALGORITHM.md](PAPER_ALGORITHM.md)；下文关于保留 ReLU、原停止条件等说明属于历史源码路径。

主复现采用官方仓库在论文发表前的最新快照 [f00df860（2024-06-03）](https://github.com/LchengC/iLIF/commit/f00df860f372eaa7fdb5924334ba0f50d41e3672)。图编码文件初次发布于2024-05-21，原实现直接传入 `source→target` 边索引。

[2024-10-29 的修改 8769cab8](https://github.com/LchengC/iLIF/commit/8769cab82f00eb7fbfe9c4223253f69832105980) 为确定性计算改用 SparseTensor，但将原 source 写入接收节点所在的 row，改变了消息方向。提供的 `reference_source` 含有这一后续改动，因此最初按该代码训练的结果只保留作版本对照。

主实现通过正确转置的稀疏邻接保持发表前传播语义，命令开关为 `--graph-orientation incoming`。`checks.py --publication` 直接导入官方快照，比较训练/评估前向及全部参数梯度，并对确定性实现执行严格 CUDA 反向检查。原版 COO 散射存在浮点加法顺序变化，测试使用固定初始种子209及声明的 float32 容差，而不是要求与原版逐位相同。

代码快照保存在 `official_publication_source`；官方提交历史、文件树、逐文件哈希和差异分别见 `artifacts/official_cge_commits.json`、`official_publication_tree.json`、`publication_version_audit.json`、`publication_vs_provided.diff`。

本地 EventStoryLine 三分类训练/开发缓存的 Git blob 哈希与发表前官方版本完全一致。这也说明两篇文档残留有向环的问题来自官方提供缓存。二分类缓存并非只改标签：原 EFFECT 正例的事件顺序被反转，训练池2674对、开发集193对；无因果关系事件对的顺序未变。该输入协议会影响二分类结果，报告保留这一数据来源差异。

发表前代码要求 torch 1.11.0，与当前 ilif 一致。Python 3.8.20/transformers 4.21.1 与原要求3.7.13/4.15.0仍有版本差异。二分类头由原固定三类改为论文定义的两类；MAVEN 沿用作者另提供的窗口/首 mention 预处理，并使用发表前图传播语义。所有差异如实保留，不据此承诺逐位复现。

公式（4）专项检查后，主模型目录改为 `esc_direction_full_incoming_paper`、`esc_existence_full_incoming_paper`、`maven_direction_full_incoming_paper`：图传播仍对照发表前快照，但融合按论文句内0.7/跨句0.3并绑定关系类型。原 `_incoming` 模型沿用作者 README 示例命令及位置融合，保留为源码对照；先前三折后续版本模型保留在 `esc_direction_full`，三种设置不能混合平均。详见 `FUSION_WEIGHT_AUDIT.md`。
