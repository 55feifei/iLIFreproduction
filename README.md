# iLIF 论文算法复现与消融实验代码

论文：[Identifying while Learning for Document Event Causality Identification](https://aclanthology.org/2024.acl-long.210/)（ACL 2024）。

本仓库整理 iLIF 文档级事件因果识别的复现代码与论文表 3 消融实验代码。**仓库只发布代码、运行说明和小型校验记录，不包含数据集、BERT 权重、训练输出、逐折结果或实验报告。**

## 算法要点

当前默认算法为 `paper_algorithm_v4`，完整规则、公式对应及论文未明确说明的取法见 [PAPER_ALGORITHM.md](PAPER_ALGORITHM.md)。

- BERT-base uncased 微调；AdamW，PLM 学习率 2e-5、其他模块 1e-4，10% 线性预热，batch size=1。
- 初始化 z0=h。训练状态的每轮迭代按 argmax 建因果边，不使用阈值 ω；开发集和测试集的迭代建图要求概率 ≥0.6。最终输出取最后一轮 argmax。
- 4 头图注意力；图编码时按论文图 2 为每个节点加句内自环，自环不计入因果图和邻接差；按关系类型及实际入边融合，句内 0.7、跨句 0.3。
- 以原文所有句子数 n 确定 min(n,L) 次图更新上限，ESC L=9，MAVEN L=19；邻接矩阵 L1 差 ≤2 停止，方向反转计 2。
- 两层 MLP、LeakyReLU、dropout=0.4。事件对 focal loss 取平均（与作者代码一致，论文公式 7 为求和），gamma=2，正类 alpha=0.75、NONE=0.25；更新后第 l 轮损失乘 1/l，初始化预测不计损失。
- 开发集标准方向/存在性 F1 选轮；方向判错同时计 FP 和 FN。

## 目录结构

- **主复现**：`run.py`（可续训训练）、`run_publication.py`（ESC/MAVEN 正式队列）、`infer.py`、`report.py`、`analyze_results.py`、`finalize_main.py`、`stress.py`。
- **模型实现**：`paper_model.py`、`adapted_model.py`、`model/`、`algorithm_config.py`、`fusion_config.py`。
- **消融实验**：`run_ablations.py`（论文表 3 完整队列，含完整模型对照）、`local_ablation_queue.py`（本机仅消融队列）、`ablation_config.py`、`ablation_preflight.py`、`check_ablations.py`、`analyze_ablations.py`、`prepare_esc_existence.py`。
- **检查与核验**：`check_paper_algorithm.py`、`check_paper_fusion.py`、`check_ablation_queue.py`、`check_local_ablation_queue.py`、`check_esc_preparation.py`、`verify_reload.py`、`checks.py`。
- **公共模块**：`common.py`、`project_paths.py`、`legacy_utils.py`、`parameter.py`。
- **文档**：[ENVIRONMENT.md](ENVIRONMENT.md)、[PAPER_ALGORITHM.md](PAPER_ALGORITHM.md)、[ABLATION_REPRODUCTION.md](ABLATION_REPRODUCTION.md)、[SERVER_ESC_RUN.md](SERVER_ESC_RUN.md)、[PUBLICATION_VERSION.md](PUBLICATION_VERSION.md)。
- **校验记录**：`artifacts/environment.txt`（本机环境快照）、`artifacts/publication_version_audit.json`（官方源码版本哈希）。其余 `artifacts/`、`runs/`、`data/`、`models/` 内容由运行产生，不随仓库分发。

## 环境与数据

推荐 Python 3.8、torch 1.11.0+cu113、transformers 4.21.1、PyG 2.1.0.post1；固定依赖见 [requirements-reproduction.txt](requirements-reproduction.txt)，本机实测配置见 [ENVIRONMENT.md](ENVIRONMENT.md)。

```bash
conda create -n ilif python=3.8 -y
conda activate ilif
python -m pip install -r requirements-reproduction.txt
```

准备以下资源（均不随仓库分发）：

- `models/bert-base-uncased/`：`config.json`、`vocab.txt`、`tokenizer.json`、`pytorch_model.bin`。`download_model.py` 可下载并校验；也可从 Hugging Face 获取。
- ESC 作者特征缓存：方向任务 train/dev 文件及 `exist/` 下的二分类 train/dev 文件。代码默认查找 `data/esc`、`data`、`../reference_source/data/database`，也可用 `ILIF_ESC_DATA` 或 `--esc-data` 指定。
- MAVEN：由 `prepare_maven.py` 从作者原始数据生成 `data/maven/train.pkl`、`dev.pkl`、`test.pkl`（首 mention、窗口 254、步长 64）。

官方源码快照 `official_publication_source/` 未随本仓库分发；如运行 `checks.py --publication` 或 `audit_publication_version.py` 的等价性检查，请从作者仓库 [LchengC/iLIF](https://github.com/LchengC/iLIF) 获取对应提交 `f00df860`（2024-06-03）。主训练不依赖该快照。

## 快速开始

在仓库根目录、已激活 ilif 环境后运行：

```bash
# 算法与融合检查（不需要数据或 GPU）
python check_paper_algorithm.py
python check_paper_fusion.py
python check_ablations.py
python check_ablation_queue.py
python check_local_ablation_queue.py
python check_esc_preparation.py

# 主实验短试跑：各训练 8 篇、评估 4 篇、1 轮
python run_publication.py --pilots-only
python verify_paper_pilots.py
```

## 主实验训练

```bash
# 正式队列：ESC 方向五折 → ESC 存在性五折 → MAVEN 方向
python run_publication.py

# 或单项运行（默认 paper_algorithm_v4）
python run.py --dataset esc --task direction --checkpoint-bert --resume
python run.py --dataset esc --task existence --checkpoint-bert --resume
python run.py --dataset maven --checkpoint-bert --resume

# 全部正式训练结束后核验并生成主表
python finalize_main.py
```

## 消融实验（论文表 3）

`run_ablations.py` 是完整队列：依赖/数据预检 → 8 项小样本试跑 → 完整模型与三个消融（2 任务 × 4 变体 × 5 折 = 40 个正式折模型，每折 30 轮）→ 逐折独立重载核验。默认只使用 ESC。

```bash
# 只打印计划，不导入训练依赖、不写文件
python run_ablations.py --dry-run --esc-data <ESC缓存目录> --model-path <BERT目录>

# 检查 CUDA、依赖和输入文件，并记录 SHA-256
python run_ablations.py --preflight-only --esc-data <ESC缓存目录> --model-path <BERT目录>

# 完整队列；仅试跑加 --pilots-only，也可用 --variants/--tasks/--folds 运行子集
python run_ablations.py --esc-data <ESC缓存目录> --model-path <BERT目录>
```

`local_ablation_queue.py` 是本机消融专用队列：不重训完整模型，只训练 `no_direction`、`no_heterogeneity`、`no_iteration`，默认输出到独立的 `runs/esc_local_ablations_v4_*` 批次目录：

```bash
python local_ablation_queue.py --dry-run
python local_ablation_queue.py --model-path <BERT目录>
```

只有方向任务缓存时，可先派生存在性缓存（EFFECT 事件对反转后合并为正类，不覆盖原方向数据）：

```bash
python prepare_esc_existence.py --esc-data <ESC缓存目录>
```

消融结果由 `analyze_ablations.py` 写入 `ABLATION_RESULTS.md`、`artifacts/ablation_comparison.csv` 和 `artifacts/verified_ablations.json`；只有五折训练、预测复算和独立重载全部通过的实验才进入正式结果。服务器后台运行与续训步骤见 [SERVER_ESC_RUN.md](SERVER_ESC_RUN.md)。

## 保存、推理与报告

每折保存 `configuration.json`、`split.json`、`history.jsonl`、`best.pt`、`last.pt`、`best_predictions.json`、`result.json` 及随机状态。续训仅从完整 epoch 恢复，并拒绝算法版本、模型、阈值、融合、损失等关键设置变化。

```bash
python infer.py --checkpoint runs/<实验名>/fold_1/best.pt --output artifacts/<输出>.json
python report.py
python analyze_results.py
```

`RESULTS.md`、`ABLATION_RESULTS.md`、`runs/`、`artifacts/` 均由运行生成，不随仓库分发。

## 复现说明

- ESC 使用作者处理后的缓存：258 篇文档、5,478 条因果关系；233 篇做文档级五折，25 篇固定开发。缓存中 `1_3`、`8_5` 两篇仍含有向环，保留原标签。
- MAVEN 使用作者原始标签及首 mention、254/64 窗口协议；原验证集 709 篇作测试。训练不截断文档或事件对。
- 张量为 FP32，不启用混合精度；保留 torch 1.11 的 TF32 开关并记录配置。BERT 梯度检查点；MAVEN 分类模块在节点表示层保存检查点并重算事件对差值，保留随机状态。
- 与作者环境存在版本差异（Python 3.8.20 / transformers 4.21.1 vs 作者 3.7.13 / 4.15.0），且论文未明确部分采用本文档声明的取法。因此本仓库保证代码可运行、结果可复核，不保证与论文数值逐位一致；详见 [PAPER_ALGORITHM.md](PAPER_ALGORITHM.md) 和 [PUBLICATION_VERSION.md](PUBLICATION_VERSION.md)。

## 引用

```bibtex
@inproceedings{liu2024ilif,
  title     = {Identifying while Learning for Document Event Causality Identification},
  author    = {Liu, Cheng and Xiang, Wei and Wang, Bang},
  booktitle = {Proceedings of ACL 2024},
  pages     = {3815--3827},
  year      = {2024},
  doi       = {10.18653/v1/2024.acl-long.210}
}
```
