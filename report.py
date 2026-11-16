"""Build an honest current report from experiment files; never invent results."""
import json
import time
from common import ROOT, save_json
from algorithm_config import PAPER, PAPER_V3, PAPER_SUFFIX, profile, validate_contract

TARGETS = {
    'esc_direction_full_incoming_paper' + PAPER_SUFFIX: ('EventStoryLine 方向', 'source_direction', [60.0, 42.8, 47.8]),
    'esc_existence_full_incoming_paper' + PAPER_SUFFIX: ('EventStoryLine 存在性', 'existence', [71.2, 59.1, 62.5]),
    'maven_direction_full_incoming_paper' + PAPER_SUFFIX: ('MAVEN-ERE 方向', 'source_direction', [59.4, 55.3, 55.6]),
}


def mean_f1(completed, mode):
    return {scope: sum(f['test'][mode][scope]['f1'] for f in completed) / len(completed) * 100
            if completed else None for scope in ('intra', 'inter', 'all')}


def format_f1(values, incomplete):
    if values['all'] is None:
        return '尚无完整训练结果'
    return '/'.join('%.2f' % values[scope] for scope in ('intra', 'inter', 'all')) + (
        '（未完成）' if incomplete else '')


def main():
    lines = ['# iLIF 复现进度与结果', '', '更新时间：' + time.strftime('%Y-%m-%d %H:%M:%S'), '',
        '论文：[Identifying while Learning for Document Event Causality Identification](https://aclanthology.org/2024.acl-long.210/)。', '',
        '论文数值来自表 1、表 2，以下实测数值只读取实际生成的结果文件。小规模试跑不代表论文结果复现成功。', '',
        '本报告汇总主实验；论文表 3 消融另见 ABLATION_RESULTS.md，运行说明见 ABLATION_REPRODUCTION.md。', '',
        '主复现采用发表前官方快照 f00df860 的图传播语义。此前按2024年10月后版本运行的结果保留为版本对照，不混入主复现。详见 PUBLICATION_VERSION.md。', '',
        '主算法为 %s，实验名带 incoming_paper%s 后缀。训练状态迭代建图不使用阈值 ω（与作者代码 --rate 默认 0 一致，按 argmax 建边）；开发集和测试集的迭代建图仍要求概率≥0.6。句内路径自环（论文图2）、最终输出取 argmax、标准方向评分、真实句子数迭代上限和邻接差停止。paper_v3 在训练和评估建图都使用 0.6，保留为对照。旧 incoming、incoming_paper、无自环的 incoming_paper_v1 及阈值输出的 incoming_paper_v2 均为历史对照，禁止混合续训。详见 PAPER_ALGORITHM.md。' % (PAPER, PAPER_SUFFIX), '',
        '| 实验 | 已完成折数 | 论文句内/跨句/总体 F1 (%) | 实测标准句内/跨句/总体 F1 (%) | 实测源码口径句内/跨句/总体 F1 (%) |',
        '|---|---:|---|---|---|']
    results = {}
    for name, (label, mode, target) in TARGETS.items():
        directory = ROOT / 'runs' / name
        if (directory / 'configuration.json').exists():
            config = json.loads((directory / 'configuration.json').read_text(encoding='utf-8'))
            validate_contract(config)
            if profile(config) != PAPER:
                raise ValueError('Main table requires %s: %s' % (PAPER, name))
            if config['partial_experiment'] or config['epochs'] != 30 or config['limit_train'] or config['limit_eval']:
                raise ValueError('Main-table directory contains a pilot: ' + name)
        completed = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(directory.glob('fold_*/result.json'))]
        expected = 1 if name.startswith('maven') else 5
        values = mean_f1(completed, mode)
        standard_mode = 'direction' if mode == 'source_direction' else 'existence'
        standard_values = mean_f1(completed, standard_mode)
        observed = format_f1(values, len(completed) < expected)
        observed_standard = format_f1(standard_values, len(completed) < expected)
        lines.append('| %s | %d/%d | %s | %s | %s |' % (label, len(completed), expected,
                     '/'.join(map(str, target)), observed_standard, observed))
        results[name] = dict(completed_folds=len(completed), expected_folds=expected,
                             paper_f1=target, measured_f1=values['all'],
                             measured_f1_protocol=mode, standard_f1=standard_values['all'],
                             measured_f1_by_scope=values, standard_f1_by_scope=standard_values)
        if name.startswith('maven'):
            existence = mean_f1(completed, 'existence')
            observed_existence = format_f1(existence, len(completed) < expected)
            lines.append('| MAVEN-ERE 存在性（由方向预测合并） | %d/%d | 60.9/56.8/57.1 | %s | %s |' %
                (len(completed), expected, observed_existence, observed_existence))
    lines += ['', '三项 F1 按句内/跨句/总体顺序列出，均为已完成折的测试集 F1 算术平均值（百分比）。', '',
              '## 训练建图阈值对照（ESC）', '',
              'v3 在训练和评估的每一轮迭代都用 ω=0.6 建图。v4 只在评估时使用该阈值，训练时按作者代码的 `--rate=0` 取 argmax 建边。两者最终输出都是 argmax。下表为已完成折，未完成时不能当作五折结果。', '',
              '| 实验 | 已完成折数 | 论文句内/跨句/总体 F1 (%) | 实测标准句内/跨句/总体 F1 (%) | 实测源码口径句内/跨句/总体 F1 (%) |',
              '|---|---:|---|---|---|']
    baselines = {
        'esc_direction_full_incoming_paper_v3': ('EventStoryLine 方向 v3（训练用 ω）', 'source_direction', [60.0, 42.8, 47.8]),
        'esc_existence_full_incoming_paper_v3': ('EventStoryLine 存在性 v3（训练用 ω）', 'existence', [71.2, 59.1, 62.5]),
        'esc_direction_full_incoming_paper' + PAPER_SUFFIX: ('EventStoryLine 方向 v4（训练不用 ω）', 'source_direction', [60.0, 42.8, 47.8]),
        'esc_existence_full_incoming_paper' + PAPER_SUFFIX: ('EventStoryLine 存在性 v4（训练不用 ω）', 'existence', [71.2, 59.1, 62.5]),
    }
    for name, (label, mode, target) in baselines.items():
        directory = ROOT / 'runs' / name
        completed = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(directory.glob('fold_*/result.json'))]
        if (directory / 'configuration.json').exists():
            config = json.loads((directory / 'configuration.json').read_text(encoding='utf-8'))
            validate_contract(config)
            expected_profile = PAPER_V3 if name.endswith('_v3') else PAPER
            if profile(config) != expected_profile:
                raise ValueError('Comparison row profile mismatch: ' + name)
        values = mean_f1(completed, mode)
        standard_mode = 'direction' if mode == 'source_direction' else 'existence'
        standard_values = mean_f1(completed, standard_mode)
        lines.append('| %s | %d/5 | %s | %s | %s |' % (label, len(completed),
                     '/'.join(map(str, target)), format_f1(standard_values, len(completed) < 5),
                     format_f1(values, len(completed) < 5)))
    lines += ['', '## 作者命令融合设置的已完成结果（对照）', '',
              '这些结果采用句内0.3/跨句0.7及位置融合，不能标成论文公式（4）设置。', '',
              '| 实验 | 已完成折数 | 标准总体 F1 (%) | 源码口径总体 F1 (%) |',
              '|---|---:|---:|---:|']
    for old in ['esc_direction_full_incoming', 'esc_existence_full_incoming', 'maven_direction_full_incoming']:
        records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'runs'/old).glob('fold_*/result.json'))]
        if records:
            task = 'existence' if 'existence' in old else 'direction'
            source = 'existence' if task == 'existence' else 'source_direction'
            lines.append('| %s | %d | %.2f | %.2f |' % (old, len(records),
                mean_f1(records, task)['all'], mean_f1(records, source)['all']))
    legacy = [json.loads(path.read_text(encoding='utf-8')) for path in
              sorted((ROOT/'runs/esc_direction_full').glob('fold_*/result.json'))]
    if legacy:
        lines += ['', '## 后续版本对照（非论文主复现）', '',
            '提供的后续版本已完成 %d 折，标准方向 F1 均值 %.2f%%；源码口径 %.2f%%。该版本反转了图消息传播方向。' %
            (len(legacy), sum(r['test']['direction']['all']['f1'] for r in legacy)/len(legacy)*100,
             sum(r['test']['source_direction']['all']['f1'] for r in legacy)/len(legacy)*100)]
    lines += ['', '## 当前运行状态', '']
    status = ROOT / ('runs/queue_status_paper%s.json' % PAPER_SUFFIX)
    if status.exists():
        state = json.loads(status.read_text(encoding='utf-8'))
        lines += ['```json', json.dumps(state, ensure_ascii=False, indent=2), '```', '']
    for path in sorted((ROOT / 'runs').glob('*/progress.json')):
        lines.append('- `%s`：%s' % (path.parent.name, path.read_text(encoding='utf-8').replace('\n', ' ')))
    lines += ['', '## 评价口径', '',
        '- `direction`：标准 CAUSE/EFFECT 微平均；方向判错同时计入 FP 和 FN。',
        '- `existence`：合并 CAUSE/EFFECT 后评价因果存在性。',
        '- `source_direction`：复刻作者 `transfor3to2`，方向判错只计 FN、不计 FP；用于核对源码日志。',
        '- %s 根据开发集最终预测（argmax）的标准方向/存在性 F1 选择最佳 epoch；源码口径仅供诊断。测试集不参与选择。' % PAPER, '',
        '## 复现限制与代码差异', '',
        '1. 使用 ilif：Python 3.8.20、torch 1.11.0+cu113、transformers 4.21.1。发表前要求 torch 1.11.0，与当前一致；Python/transformers 版本仍不同，不承诺逐位相同。',
        '2. EventStoryLine 直接使用作者处理后的缓存，258 篇文档、5,478 条因果关系；233 篇做文档级五折，25 篇作固定开发集。',
        '作者缓存中 1_3、8_5 两篇文档仍有有向环，与论文预处理后为 DAG 的描述不完全一致；保留原标签，详见 artifacts/data_integrity.json。',
        '3. %s 使用 LeakyReLU、句内0.7/跨句0.3类型融合、z0=h、至多min(全文句数,L)次有效图更新及邻接矩阵差停止。图编码时按论文图2为每个节点加句内自环，自环不计入因果图和邻接差；不做空图投影。训练状态迭代建图不使用 ω；评估状态的迭代建图和 final_ecg 仍用≥0.6。最终输出取最后一轮 argmax。初始预测不计损失；更新后按1/l加权事件对损失之和。未明确的取法见 PAPER_ALGORITHM.md。' % PAPER,
        '4. 为适配 8GB 显存启用 BERT 梯度检查点，MAVEN 还对分类模块使用保留随机状态的梯度检查点。focal loss 的 gather 替换为等价索引以支持当前 torch 的严格确定性反向。图构建与差分已向量化，CUDA 上的前向等价与严格确定性反向已测试。使用 FP32 张量、不启用混合精度，保留 torch 1.11 默认允许 TF32 的矩阵计算开关，不截断文档或事件对；精度探测见 artifacts/numeric_precision.json。',
        '5. MAVEN 沿用作者每个事件只选第一个 mention、文档内一半正例反向的策略；修正打乱事件对之后句内/跨句标记的错位及完整触发词在窗口边界处被遗漏的问题。窗口长度 254、步长 64。这些修正可能导致与作者原日志不同。',
        '6. MAVEN 原 train 划分 2621/292，原 valid 中 709 篇至少含两个事件的文档用于测试；原 test 无标签，未用于报告测试性能。', '',
        '详见 README.md 的命令与产物说明。']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save_json(ROOT / 'artifacts/result_comparison.json', results)


if __name__ == '__main__':
    main()
