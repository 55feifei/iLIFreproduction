# iLIF 复现代码运行环境

适用范围：本仓库中的论文算法复现代码（当前默认 `paper_algorithm_v4`）。以下为本机实测配置；固定依赖见 [requirements-reproduction.txt](requirements-reproduction.txt)，完整环境快照见 [artifacts/environment.txt](artifacts/environment.txt)。

## 1. 硬件与系统

| 项目 | 实测配置 | 说明 |
|---|---|---|
| 操作系统 | Windows 11 家庭中文版（10.0.26300） | 训练也可在 Linux 服务器运行 |
| CPU | Intel Core i5-13500HX，14 核 / 20 线程 | 训练以 GPU 为主 |
| 内存 | 16 GB | |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU | 8187 MiB 显存，compute capability 8.9，24 SM |
| 显卡驱动 | NVIDIA 621.14（CUDA UMD 13.5） | 无需单独安装 CUDA Toolkit；PyTorch 1.11 的 cu113 运行库经驱动向后兼容运行 |
| 磁盘 | 实测可用 251 GB，建议预留 ≥30 GB | 每折 `best.pt`≈441 MB、`last.pt`≈1.3 GB，完整 11 个模型约 20 GB |

## 2. Python 环境与核心依赖

- Python 3.8.20（Anaconda），实测解释器：`F:\ANACONDA\envs\ilif\python.exe`
- 已在本机实际导入验证：

| 依赖 | 版本 |
|---|---|
| torch | 1.11.0+cu113（CUDA 可用） |
| transformers | 4.21.1 |
| tokenizers | 0.12.1 |
| torch-geometric | 2.1.0.post1 |
| torch-scatter | 2.0.9 |
| torch-sparse | 0.6.13 |
| numpy | 1.22.4 |
| scipy | 1.10.1 |
| scikit-learn | 1.0.2 |
| matplotlib | 3.5.3 |
| networkx | 3.1 |
| tqdm | 4.67.1 |
| huggingface-hub | 0.34.4 |

`artifacts/environment.txt` 是共享 conda 环境的全量导出，含 LIBERO、mujoco 等与本项目无关的包，只作快照，不要照单安装；应以 `requirements-reproduction.txt` 为准。

代码版本：本仓库根目录即复现代码目录，包含主复现、论文表 3 消融队列和检查脚本；官方源码快照不随仓库分发，主算法对照官方提交 `f00df860`（2024-06-03）的发表前版本。

## 3. 安装步骤

```powershell
conda create -n ilif python=3.8 -y
conda activate ilif
cd <仓库根目录>
python -m pip install -r requirements-reproduction.txt
```

`requirements-reproduction.txt` 已写入 PyTorch cu113 源和 PyG 对应轮子的查找地址。`torch-scatter`、`torch-sparse` 必须与 `torch==1.11.0+cu113` 匹配，否则 PyG 导入会失败。

安装后验证：

```powershell
python -c "import torch, transformers, torch_geometric; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, torch_geometric.__version__)"
# 预期：1.11.0+cu113 True 4.21.1 2.1.0
```

## 4. 预训练权重与数据

| 资源 | 位置 | 说明 |
|---|---|---|
| BERT-base-uncased | `models/bert-base-uncased/` | `config.json`、`vocab.txt`、`tokenizer.json`、`pytorch_model.bin`（约 440 MB）。来源 `https://huggingface.co/bert-base-uncased`；`download_model.py` 可下载并校验 SHA256 `097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a`，支持 hf-mirror 镜像 |
| ESC 作者缓存 | `..\reference_source\data\database\` | 方向任务 `*_train/dev_features.pkl`，存在性任务在 `exist/` 子目录。233 篇文档做五折、25 篇固定开发；缓存与发表前官方版本 Git blob 哈希一致 |
| MAVEN 预处理数据 | `data/maven/` | `train.pkl`（2621 篇）、`dev.pkl`（292 篇）、`test.pkl`（709 篇，原验证集作测试）。由 `prepare_maven.py` 从作者 MAVEN 原始数据生成，窗口 254、步长 64、首 mention；`manifest.json` 记录参数与 SHA256 |
| 官方源码快照 | 未随仓库分发 | 对应官方提交 `f00df860`（2024-06-03），仅用于等价性检查，当前主训练不直接调用；如运行 `checks.py --publication`，请从作者仓库获取该提交 |

代码包本身不含 BERT 权重、ESC 缓存和 MAVEN 原始数据，换机复现时需一并拷贝或重新下载/预处理。

## 5. 运行前自检

在仓库根目录执行（以下 python 路径为本机实测值，换机时替换）：

```powershell
& F:\ANACONDA\envs\ilif\python.exe check_paper_algorithm.py
& F:\ANACONDA\envs\ilif\python.exe check_paper_fusion.py
& F:\ANACONDA\envs\ilif\python.exe run_publication.py --pilots-only
& F:\ANACONDA\envs\ilif\python.exe verify_paper_pilots.py
```

前两项应输出 `"status": "passed"`。`--pilots-only` 为三个短试跑（ESC 方向、ESC 存在性、MAVEN 方向），各训练 8 篇、评估 4 篇、1 轮，不会自动进入正式训练；检查证据写入 `artifacts/paper_algorithm_checks.json` 和 `artifacts/paper_algorithm_pilots.json`。

## 6. 训练、推理与报告

```powershell
# 正式队列：ESC 方向五折 → ESC 存在性五折 → MAVEN 方向
& F:\ANACONDA\envs\ilif\python.exe run_publication.py

# 或单项运行（默认 paper_algorithm_v4）
& F:\ANACONDA\envs\ilif\python.exe run.py --dataset esc --task direction --checkpoint-bert --resume
& F:\ANACONDA\envs\ilif\python.exe run.py --dataset esc --task existence --checkpoint-bert --resume
& F:\ANACONDA\envs\ilif\python.exe run.py --dataset maven --checkpoint-bert --resume

# 全部训练结束后核验并生成主表
& F:\ANACONDA\envs\ilif\python.exe finalize_main.py

# 单模型推理与报告
& F:\ANACONDA\envs\ilif\python.exe infer.py --checkpoint runs\esc_direction_full_incoming_paper_v3\fold_1\best.pt --output artifacts\paper_fold1_inference.json
& F:\ANACONDA\envs\ilif\python.exe report.py
& F:\ANACONDA\envs\ilif\python.exe analyze_results.py
```

## 7. 资源占用与注意事项

- 训练使用 FP32，不启用混合精度；保留 torch 1.11 的 TF32 开关。batch size = 1，一个样本是一篇文档，文档内部包含多个事件对。
- 本机实测（ESC）：训练加开发评估约 42–44 秒/轮，五折约 112–116 分钟/任务，日志峰值 allocated 显存 2.91 GB。该数字未包含全部测试、保存与报告开销，也不是整卡显存占用；MAVEN 文档更长（图更新上限 L=19），显存与耗时更高，正式训练前可用 `stress.py --dataset esc` 或 `--dataset maven` 检查最大文档。
- 8 GB 显存可满足本机 ESC 训练；16 GB 内存为实测配置。CPU 能运行检查与短试跑，但正式训练不建议。
- 与作者环境存在版本差异：当前 Python 3.8.20 / transformers 4.21.1 / torch 1.11.0+cu113，作者要求 Python 3.7.13 / transformers 4.15.0 / torch 1.11.0。这些差异、FP32/TF32 设置及论文未明确细节会限制逐位数值复现；本环境保证代码可运行、结果可核验，不保证与论文主表逐位一致，详见 [PUBLICATION_VERSION.md](PUBLICATION_VERSION.md)。

## 8. 最短复现路径

1. 按第 3 节创建环境并安装固定依赖。
2. 确认 `models/bert-base-uncased/` 与第 4 节数据就绪。
3. 依次通过第 5 节四项自检。
4. 运行 `run_publication.py` 完成正式训练。
5. 运行 `finalize_main.py`，生成主表与核验证据。
