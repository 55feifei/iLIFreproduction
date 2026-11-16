# 论文表 3 消融实验：服务器运行说明

对应论文 [Identifying while Learning for Document Event Causality Identification](https://aclanthology.org/2024.acl-long.210.pdf) 第 4.4 节、表 3。基于项目当前 `paper_algorithm_v4`，三个消融只修改相应图机制，保留其余训练、评估及选轮设置。实现解释见 `PAPER_ALGORITHM.md`。

## 实验范围

| 参数值 | 图编码方式 | Direction 论文 P/R/F1 (%) | Existence 论文 P/R/F1 (%) |
|---|---|---|---|
| `none` | 完整有向异构图，迭代更新 | 47.9/47.8/47.8 | 59.2/66.1/62.5 |
| `no_direction` | 因果边双向化，保留句内/跨句类型 | 42.3/49.3/45.5 | 55.9/63.8/59.6 |
| `no_heterogeneity` | 合并为一个关系类型，使用同一注意力分支 | 45.5/47.2/46.3 | 56.4/66.3/61.0 |
| `no_iteration` | 初始分类建图 → 一次图编码 → 最终分类 | 43.2/45.2/44.2 | 52.8/65.9/58.6 |

默认包括完整模型对照，共 4 个变体 × 2 种任务 × 5 折 = **40 个正式模型**，各训练 30 个 epoch。方向任务保留 NONE/CAUSE/EFFECT 三类，即使其图移除了方向；存在性任务使用单独二分类训练。表 3 只在 EventStoryLine 上做消融。

按项目协议使用 seed=209、文档级 KFold、固定开发集；保留作者缓存的标签和事件对顺序。训练使用 argmax 建图，评估迭代建图阈值为 0.6，最终输出为 argmax。无方向图的相邻图差以双向邻接矩阵计算，单条无向边的增删计 2；单次更新变体仅最终预测计入 loss，CGE 仍参与反向传播。

服务器上的上传、解压、后台训练和日志命令见 [SERVER_ESC_RUN.md](SERVER_ESC_RUN.md)。可用 `prepare_esc_existence.py --archive artifacts/esc_data.zip` 生成单独的 ESC 数据包，再解压到项目 `data/` 下，得到 `data/esc/`。

如果只有方向任务的两份缓存，可运行 `python prepare_esc_existence.py --esc-data data --archive artifacts/esc_data.zip`：按附录 B 翻转 EFFECT 事件对并生成二分类缓存，记录来源，不覆盖原方向数据。派生缓存不是作者原二分类文件的字节副本。

## 准备服务器

建议使用与项目一致的 Python 3.8、torch 1.11.0+cu113、transformers 4.21.1、PyG 2.1.0.post1 环境，以及兼容 CUDA 的 NVIDIA GPU。依赖固定在 `requirements-reproduction.txt`。在服务器项目根目录运行：

```bash
conda create -n ilif-ablation python=3.8 -y
conda activate ilif-ablation
python -m pip install -r requirements-reproduction.txt
```

将本仓库源码迁移到服务器项目根目录。源码不包含本机临时测试依赖、数据、预训练权重和历史训练输出。

准备本地 `bert-base-uncased` 的 `config.json` 和 `pytorch_model.bin`，以及以下 **4 份 ESC 缓存**。`--esc-data` 指向同时包含方向缓存和 `exist/` 的根目录：

```text
/server/data/esc/
  EventStoryLine_bert_intra_and_inter_train_features.pkl
  EventStoryLine_bert_intra_and_inter_dev_features.pkl
  exist/
    EventStoryLine_bert_intra_and_inter_train_features.pkl
    EventStoryLine_bert_intra_and_inter_dev_features.pkl
```

训练池应为 233 篇文档，固定开发集为 25 篇。二分类缓存沿用项目存在性输入协议：EFFECT 正例的事件对顺序翻转，再标为 CAUSE；不能只将标签 2 改为 1。服务器已有资源时直接指定路径。程序导入不再依赖原作者中文目录或旧 F 盘路径。

## 检查与启动

先检查命令计划，此命令无需训练依赖，也不会创建实验文件：

```bash
python run_ablations.py --dry-run \
  --esc-data /server/data/esc \
  --model-path /server/models/bert-base-uncased
```

预检检查 CUDA、依赖导入和文件存在性，并记录输入 SHA-256：

```bash
python run_ablations.py --preflight-only \
  --esc-data /server/data/esc \
  --model-path /server/models/bert-base-uncased
```

启动完整队列：

```bash
python -u run_ablations.py \
  --esc-data /server/data/esc \
  --model-path /server/models/bert-base-uncased
```

程序串行执行：队列与模型语义检查 → 全部 8 项小样本试跑及重载核验 → 8 项正式训练及逐折重载核验 → 汇总报告。试跑各训练 8 篇、评估 4 篇、1 个 epoch。任一阶段失败即停止，错误保存在对应日志；全部试跑成功后才开始正式训练。

仅试跑加 `--pilots-only`。重复执行相同命令会通过 `--resume` 从最近完整 epoch 恢复；完成的折跳过训练，但重新核验最佳模型。不要同时向同一输出目录运行多个队列。检查点及优化器状态会占用较多磁盘空间。

默认输出到 `runs/`；`--output-root /server/experiments/ilif-ablations` 可指定位置。可用 `--variants no_direction no_heterogeneity no_iteration` 只训练消融，或用 `--tasks direction` 只运行方向任务。相对完整模型的 ΔF1 要求完整模型位于同一输出根目录，且 seed、数据、预训练权重、精度设置和折划分匹配，并完成重载核验。

`--folds 1 2` 可先运行部分正式折，只表示进度，不会发布为完整五折结果。复用已有主实验前需核对版本及配置；不要重命名旧版本结果，也不要跨变体复用检查点。

## 输出与复核

- `runs/ablation_plan_paper_v4.json`：实验计划及命令。
- `runs/ablation_queue_status_paper_v4.json`：当前实验、阶段、成功列表或失败原因。
- `runs/<实验名>.log`：训练、推理和核验日志。
- `runs/<实验名>/fold_N/`：划分、训练历史、最佳/最近检查点、预测、结果及重载证据。
- `artifacts/ablation_preflight.json`、`artifacts/ablation_checks.json`：环境检查与模型检查结果。
- `ABLATION_RESULTS.md`：论文/实测 P/R/F1、训练及重载进度、相对完整模型的 ΔF1。
- `artifacts/ablation_comparison.csv`：只收录五折训练、预测复算及独立重载全部通过的正式行。
- `artifacts/verified_ablations.json`：逐折指标与核验状态；学习曲线位于 `artifacts/charts/`。

训练期间可重新生成报告：

```bash
python analyze_ablations.py
# 自定义输出根目录时：
python analyze_ablations.py --runs-root /server/experiments/ilif-ablations
```

方向任务采用标准 CAUSE/EFFECT 微平均，方向判错同时计 FP 和 FN；开发集标准 F1 选轮，F1 持平时选较晚 epoch。跨折按每折 F1 算术平均，平均 P/R 后重算 F1 仅作诊断。ΔF1=变体 F1−完整模型 F1，单位为百分点。

重载核验比较逐文档分类、置信度（容差 1e-5）、图边和全部指标，并记录检查点、原预测、配置和划分哈希。文件改变后旧证据失效。单独重载时，`infer.py --model-path` 和 `--esc-data` 可指定迁移后的资源位置。报告读取训练配置中的数据位置；旧配置可使用 `ILIF_ESC_DATA` 环境变量。

## 验证范围

本仓库只发布代码，不包含正式训练结果。`check_ablation_queue.py` 无需训练环境；`check_ablations.py` 使用小型测试文本编码器，运行真实图编码器、MLP、focal loss、前向/反向及保存重建检查，也检查未重载结果不得进入正式 CSV。它不下载 BERT 或产生实验指标。

正式数值需要按上述队列在真实数据上完成训练；论文表 3 的数值只作参照，不是本仓库的实测结果。运行结束后，汇总输出为 `ABLATION_RESULTS.md`、`artifacts/ablation_comparison.csv` 和 `artifacts/verified_ablations.json`。
