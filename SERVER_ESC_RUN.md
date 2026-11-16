# 服务器：上传 ESC 并在本机关机后继续训练

服务器：`<user>@<server>`，SSH 端口 `<port>`。以下操作由你在终端中执行；本项目没有自动连接服务器或开始训练。

## 1. 本机 PowerShell：上传 ESC 数据

```powershell
scp -P <port> "<本地目录>\artifacts\esc_data.zip" <user>@<server>:/home/<user>/
ssh -p <port> <user>@<server>
```

`esc_data.zip` 只包含 ESC 方向/存在性缓存和派生记录，不包含 MAVEN。原始方向缓存与项目保存的官方 Git blob 哈希一致；存在性缓存由 `prepare_esc_existence.py` 按论文附录 B 派生，保留事件对行序，翻转 EFFECT 正例，再合并为正类。这是明确记录的派生数据，不宣称与作者原二分类缓存逐字节一致。

## 2. 服务器：获取代码与解压数据

```bash
cd /home/<user>
git clone https://github.com/55feifei/iLIFreproduction.git iLIF_reproduction
mkdir -p iLIF_reproduction/data
unzip -o esc_data.zip -d iLIF_reproduction/data/
cd iLIF_reproduction
```

如果服务器无法访问 GitHub，可在本机打包源码后用 `scp`/`rsync` 上传，解压到 `/home/<user>/iLIF_reproduction`。`git pull` 或代码更新不会清空 `runs/`；已有训练正在使用这些文件时，应在训练结束后更新。

数据位置：`/home/<user>/iLIF_reproduction/data/esc/`，其下有方向任务的 train/dev 文件，以及 `exist/` 下的二分类 train/dev 文件。代码已经兼容数据路径，无需手动修改训练代码。

## 3. 服务器：首次配置环境与 BERT

已有兼容环境时直接激活。首次配置可按项目固定依赖安装：

```bash
conda create -n ilif-ablation python=3.8 -y
conda activate ilif-ablation
python -m pip install -r requirements-reproduction.txt
nvidia-smi
```

本仓库和 ESC 数据包都不含 BERT 权重。服务器已有 `bert-base-uncased` 时，在后续命令中使用其实际路径；否则可在服务器联网下载一次：

```bash
python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased').save_pretrained('models/bert-base-uncased')"
```

完成后应存在 `models/bert-base-uncased/config.json` 和 `models/bert-base-uncased/pytorch_model.bin`。这是下载预训练模型，不是训练消融模型。

## 4. 服务器：先预检

```bash
python run_ablations.py --preflight-only \
  --esc-data /home/<user>/iLIF_reproduction/data/esc \
  --model-path /home/<user>/iLIF_reproduction/models/bert-base-uncased
```

看到 `"status": "passed"` 后再启动。否则按输出的 `errors` 处理缺失依赖、CUDA或文件。

## 5. 服务器：用 nohup 在后台启动

确认已进入项目根目录并激活正确环境，然后执行：

```bash
cd /home/<user>/iLIF_reproduction
conda activate ilif-ablation
nohup python -u run_ablations.py \
  --esc-data /home/<user>/iLIF_reproduction/data/esc \
  --model-path /home/<user>/iLIF_reproduction/models/bert-base-uncased \
  > esc_ablation.log 2>&1 < /dev/null &
echo $! > esc_ablation.pid
```

默认只做 ESC，包括完整模型及三个消融、方向和存在性两种任务、五折×30轮，共 40 个正式模型。队列会先执行检查和全部小样本试跑，成功后再进入正式训练。不要重复执行启动命令来创建多个同时写同一目录的队列。

确认进程存在并查看启动日志：

```bash
ps -p "$(cat esc_ablation.pid)" -o pid,etime,cmd
tail -n 50 esc_ablation.log
```

**此时可以退出 SSH、关闭终端和关闭你的电脑。训练在服务器上运行，不依赖你的电脑保持在线。** 服务器必须保持开机；服务器重启、作业被终止或程序报错仍会导致任务停止。

## 6. 随时重新登录并查看进度

本机 PowerShell：

```powershell
ssh -p <port> <user>@<server>
```

服务器：

```bash
cd /home/<user>/iLIF_reproduction
tail -f esc_ablation.log
```

按 `Ctrl+C` 退出日志查看，不会停止后台训练。队列日志会指明当前实验的详细日志位置，例如 `runs/esc_direction_full_incoming_paper_v4.log`；可以用 `tail -f` 查看该文件中的 epoch、loss 和 F1。

使用正确 Python 环境查看队列状态及 GPU：

```bash
conda activate ilif-ablation
python -m json.tool runs/ablation_queue_status_paper_v4.json
nvidia-smi
```

队列状态为 `complete` 才表示所选队列完成；`failed` 时查看对应日志。仅看到后台 PID 不代表训练已通过预检。

若任务停止，先确认旧进程已退出、修复报错，再用同样的 nohup 命令续训。完成的折会跳过训练，未完成的折从最近完整 epoch 检查点恢复。

## 7. 结果位置

`ABLATION_RESULTS.md` 是对照报告；`artifacts/ablation_comparison.csv` 是正式结果表；`runs/` 包含训练历史、模型、预测和重载证据。可重新生成报告：

```bash
python analyze_ablations.py
```

本机 PowerShell 下载结果：

```powershell
scp -P <port> <user>@<server>:/home/<user>/iLIF_reproduction/ABLATION_RESULTS.md "<本地目录>\artifacts\server_ABLATION_RESULTS.md"
scp -P <port> <user>@<server>:/home/<user>/iLIF_reproduction/artifacts/ablation_comparison.csv "<本地目录>\artifacts\server_ablation_comparison.csv"
```
