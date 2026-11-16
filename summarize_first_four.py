"""Export the completed first four ESC direction folds without partial fold 5."""
import csv
import json
import time
import numpy as np
from analyze_results import inspect_experiment
from common import ROOT, save_json


def main():
    directory = ROOT / 'runs/esc_direction_full_incoming'
    experiment = inspect_experiment(directory)
    folds = [fold for fold in experiment['verified_folds'] if fold['fold'] in [1, 2, 3, 4]]
    assert [fold['fold'] for fold in folds] == [1, 2, 3, 4]
    average = {mode: {scope: {key: float(np.mean([fold['test'][mode][scope][key] for fold in folds]))
                 for key in ['p', 'r', 'f1']} for scope in ['intra', 'inter', 'all']}
                 for mode in ['direction', 'source_direction']}
    totals = dict(documents=0, pairs=0, positive_pairs=0)
    rows = []
    for fold in folds:
        folder = directory / ('fold_%d' % fold['fold'])
        split = json.loads((folder / 'split.json').read_text(encoding='utf-8'))
        count = len(split['test'])
        gold = fold['test']['direction']['all']['gold']
        predictions = json.loads((folder / 'best_predictions.json').read_text(encoding='utf-8'))
        totals['documents'] += count
        totals['positive_pairs'] += gold
        totals['pairs'] += sum(len(record['gold']) for record in predictions)
        fold['test_documents'] = count
        for scope in ['intra', 'inter', 'all']:
            row = dict(fold=fold['fold'], best_epoch=fold['best_epoch'], scope=scope,
                       test_documents=count, gold_positive_pairs=fold['test']['direction'][scope]['gold'])
            for mode, prefix in [('direction', 'standard'), ('source_direction', 'author')]:
                row.update({prefix + '_' + key: fold['test'][mode][scope][key] * 100 for key in ['p', 'r', 'f1']})
            rows.append(row)
    save_json(ROOT / 'artifacts/first_four_results.json', dict(
        created=time.strftime('%Y-%m-%d %H:%M:%S'), experiment=directory.name,
        scope='completed folds 1-4 only; not the complete five-fold paper result',
        folds=folds, mean_fold=average, totals=totals,
        verification='recomputed saved predictions and canonical data; independent model reload pending final queue verification'))
    with (ROOT / 'artifacts/first_four_results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# EventStoryLine 方向实验：前四折结果', '',
        '整理时间：' + time.strftime('%Y-%m-%d %H:%M:%S'), '',
        '本报告固定统计发表前传播版本的前四折，每折均完成30轮训练。第五折尚未完成，其阶段性结果不纳入本报告。', '',
        'F1、P、R 均以百分比表示。最佳轮次由固定开发集选择，测试集未用于选轮。', '',
        '## 各折总体指标', '',
        '| 折 | 最佳轮次 | 测试文档 | 测试正例 | 标准 P | 标准 R | 标准 F1 | 源码口径 F1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for fold in folds:
        value = fold['test']['direction']['all']
        lines.append('| %d | %d | %d | %d | %.2f | %.2f | %.2f | %.2f |' % (
            fold['fold'], fold['best_epoch'], fold['test_documents'], value['gold'],
            value['p'] * 100, value['r'] * 100, value['f1'] * 100,
            fold['test']['source_direction']['all']['f1'] * 100))
    value = average['direction']['all']
    lines += ['| 四折均值 | — | — | — | %.2f | %.2f | **%.2f** | **%.2f** |' % (
        value['p'] * 100, value['r'] * 100, value['f1'] * 100,
        average['source_direction']['all']['f1'] * 100), '',
        '均值为各折指标的算术平均；不是将平均 P/R 重新计算得到的 F1。', '',
        '## 句内、跨句与总体', '',
        '| 范围 | 标准 P | 标准 R | 标准 F1 | 源码 P | 源码 R | 源码 F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for scope, label in [('intra', '句内'), ('inter', '跨句'), ('all', '总体')]:
        values = [average[mode][scope][key] * 100 for mode in ['direction', 'source_direction']
                  for key in ['p', 'r', 'f1']]
        lines.append('| %s | %s |' % (label, ' | '.join('%.2f' % value for value in values)))
    lines += ['',
        '方向判错在标准指标中同时计为 FP 和 FN；作者源码漏计此类 FP，因此单独列出源码口径。上述指标按分类 argmax 输出计算，0.6置信度因果图指标另见完整复核报告。', '',
        '## 覆盖与核验', '',
        '前四折共覆盖 %d 篇互不重复的测试文档、%d 个事件对和 %d 个因果正例。已核对30轮记录、开发集选轮、原始标签/事件对、跨折测试文档隔离，并从保存预测重算指标和因果图。模型独立重载核验将在完整主队列结束后执行。' %
        (totals['documents'], totals['pairs'], totals['positive_pairs']), '',
        '论文总体方向 F1 为47.8%，对应完整五折。当前四折均值仅为阶段性结果，不能作为完整论文结果复现结论。EventStoryLine 存在性和 MAVEN-ERE 正式实验尚待后续队列执行。', '',
        '暂停时第五折已保存至第16轮；收到继续训练指令后已从第17轮恢复。续训状态与暂停记录见 artifacts/pause_resume_state.json、artifacts/user_pause_confirmation.json。', '',
        '## 训练曲线', '',
        '仅绘制这四折的训练损失与开发集指标；开发集 F1 使用作者源码口径。', '',
        '![前四折训练曲线](F:/iLIF/reproduction/artifacts/charts/esc_direction_first_four_learning.png)', '',
        '数据文件：[CSV](F:/iLIF/reproduction/artifacts/first_four_results.csv)、[JSON](F:/iLIF/reproduction/artifacts/first_four_results.json)。', '',
        '更多汇总方式、逐折因果图指标和版本对照见 [完整复核报告](F:/iLIF/reproduction/VERIFIED_RESULTS.md)。', '']
    (ROOT / 'FIRST_FOUR_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for fold in folds:
        history = [json.loads(line) for line in (directory / ('fold_%d/history.jsonl' % fold['fold'])).read_text(encoding='utf-8').splitlines()]
        epochs = [record['epoch'] for record in history]
        label = 'fold_%d' % fold['fold']
        axes[0].plot(epochs, [record['train_loss'] for record in history], label=label)
        axes[1].plot(epochs, [record['validation']['source_direction']['all']['f1'] * 100 for record in history], label=label)
    axes[0].set(xlabel='Epoch', ylabel='Training loss')
    axes[1].set(xlabel='Epoch', ylabel='Development F1 (author metric, %)', ylim=(0, 100))
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.suptitle('EventStoryLine direction: completed folds 1-4')
    chart = ROOT / 'artifacts/charts/esc_direction_first_four_learning'
    fig.savefig(str(chart) + '.png', dpi=180)
    fig.savefig(str(chart) + '.pdf')
    plt.close(fig)
    print(dict(completed_folds=4, standard_f1=average['direction']['all']['f1'] * 100,
               author_f1=average['source_direction']['all']['f1'] * 100, **totals))


if __name__ == '__main__':
    main()
