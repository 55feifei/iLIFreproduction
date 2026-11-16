"""Validate completed experiment artifacts and export comparison tables/curves.

An unfinished fold is never counted as a completed experiment. Recompute metrics
from saved predictions instead of trusting summary/checkpoint labels alone.
"""
import csv
import json
from pathlib import Path
import numpy as np
from common import ROOT, ESC_DATA, load_features, save_json, scores
from algorithm_config import (PAPER as PAPER_PROFILE, PAPER_V3, PAPER_V2, PAPER_V1, PAPER_SUFFIX,
                              profile, selection_metric, validate_contract)
from project_paths import esc_data_path

PAPER = {
    ('esc', 'direction'): {'intra': [66.7,54.5,60.0], 'inter': [41.2,44.6,42.8], 'all': [47.9,47.8,47.8]},
    ('esc', 'existence'): {'intra': [76.8,66.3,71.2], 'inter': [53.5,65.9,59.1], 'all': [59.2,66.1,62.5]},
    ('maven', 'direction'): {'intra': [73.7,49.7,59.4], 'inter': [66.3,47.5,55.3], 'all': [66.9,47.6,55.6]},
    ('maven', 'existence'): {'intra': [74.4,51.5,60.9], 'inter': [67.1,49.2,56.8], 'all': [67.7,49.4,57.1]},
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_ecg(record, ablation='none'):
    assert len(record['confidence']) == len(record['pred'])
    assert all(0. <= confidence <= 1. for confidence in record['confidence'])
    # Model probabilities and the scalar threshold are compared in float32.
    # JSON decoding produces Python floats, whose representation of .6 differs.
    threshold = float(np.float32(.6))
    protocol = record.get('prediction_protocol')
    paper = protocol in ('thresholded_final_ecg', 'final_argmax')
    labels = record.get('argmax_pred', record['pred'])
    gated = [label if (confidence >= threshold if paper else confidence > threshold) else 0
             for label, confidence in zip(labels, record['confidence'])]
    if paper:
        assert record['relation_threshold'] == .6
    if protocol == 'thresholded_final_ecg':
        assert gated == record['pred'], ('Final ECG predictions were not thresholded', record['doc_id'])
    elif protocol == 'final_argmax':
        assert labels == record['pred'], ('Final predictions are not argmax', record['doc_id'])
    expected_graph = {}
    for kind, name in [(0, 'intra'), (1, 'inter')]:
        edges = []
        for label in [1, 2]:
            for pair, predicted, relation_type in zip(record['event_pairs'], gated, record['rel_type']):
                if predicted == label and relation_type == kind:
                    edges.append(pair if label == 1 else pair[::-1])
        if edges:
            expected_graph[name] = edges
    if ablation == 'no_direction':
        expected_graph = {kind: edges + [pair[::-1] for pair in edges]
                          for kind, edges in expected_graph.items()}
    elif ablation == 'no_heterogeneity' and expected_graph:
        expected_graph = {'intra': sum(expected_graph.values(), [])}
    assert expected_graph == record['final_ecg'], ('ECG/label/confidence mismatch', record['doc_id'])
    return gated


def inspect_experiment(directory):
    config = read_json(directory / 'configuration.json')
    validate_contract(config)
    expected = 5 if config['dataset'] == 'esc' else 1
    if config['dataset'] == 'esc':
        base = esc_data_path(config.get('esc_data'))
        data_dir = base / 'exist' if config['task'] == 'existence' else base
        pool = load_features(data_dir / 'EventStoryLine_bert_intra_and_inter_train_features.pkl')
        reference = {str(f.doc_id): f for f in pool}
        expected_test_ids = set(reference)
    else:
        test = load_features(ROOT / 'data/maven/test.pkl')
        reference = {str(f.doc_id): f for f in test}
        expected_test_ids = set(reference)
    result = dict(configuration=config, expected_folds=expected, verified_folds=[], complete=False,
                  ecg_direction_metrics_applicable=config.get('ablation', 'none') != 'no_direction')
    all_gold, all_pred, all_types, test_ids, all_ecg_pred = [], [], [], [], []
    fold_scores, fold_ecg_scores = [], []
    for fold_path in sorted(directory.glob('fold_*')):
        if not (fold_path / 'result.json').exists():
            continue
        saved = read_json(fold_path / 'result.json')
        split = read_json(fold_path / 'split.json')
        predictions = read_json(fold_path / 'best_predictions.json')
        history = [json.loads(line) for line in (fold_path / 'history.jsonl').read_text(encoding='utf-8').splitlines()]
        epochs = [record['epoch'] for record in history]
        assert epochs == list(range(1, config['epochs'] + 1)), ('missing/duplicate epochs', fold_path)
        assert set(split['train']).isdisjoint(split['dev'])
        assert set(split['train']).isdisjoint(split['test'])
        assert set(split['dev']).isdisjoint(split['test'])
        assert [record['doc_id'] for record in predictions] == split['test']
        assert len(set(split['test'])) == len(split['test'])
        gold, pred, types, ecg_pred = [], [], [], []
        for record in predictions:
            if profile(config) in (PAPER_PROFILE, PAPER_V3, PAPER_V2, PAPER_V1):
                assert record.get('prediction_protocol') == config['prediction_protocol'], fold_path
            feature = reference[record['doc_id']]
            assert record['gold'] == list(map(int, feature.target)), ('gold changed', record['doc_id'])
            assert record['rel_type'] == list(map(int, feature.rel_type)), ('relation types changed', record['doc_id'])
            assert record['event_pairs'] == [list(map(int, pair)) for pair in feature.event_pairs]
            assert len(record['gold']) == len(record['pred']) == len(record['rel_type']) == len(record['event_pairs'])
            gold.extend(record['gold'])
            pred.extend(record['pred'])
            types.extend(record['rel_type'])
            ecg_pred.extend(verify_ecg(record, config.get('ablation', 'none')))
        computed = scores(gold, pred, types)
        computed_ecg = scores(gold, ecg_pred, types)
        if not result['ecg_direction_metrics_applicable']:
            computed_ecg = {'existence': computed_ecg['existence']}
        for mode in computed:
            for scope in computed[mode]:
                for metric, value in computed[mode][scope].items():
                    assert abs(value - saved['test'][mode][scope][metric]) < 1e-12, (fold_path, mode, scope, metric)
        selection = selection_metric(config)
        selected = max(history, key=lambda record: (record['validation'][selection]['all']['f1'], record['epoch']))
        assert saved['epoch'] == selected['epoch'], ('epoch selection mismatch', fold_path)
        assert (fold_path / 'best.pt').stat().st_size > 400_000_000
        result['verified_folds'].append(dict(fold=saved['fold'], best_epoch=saved['epoch'],
                                            test=computed, test_ecg=computed_ecg))
        fold_scores.append(computed)
        fold_ecg_scores.append(computed_ecg)
        all_gold.extend(gold)
        all_pred.extend(pred)
        all_types.extend(types)
        all_ecg_pred.extend(ecg_pred)
        test_ids.extend(split['test'])
    assert len(test_ids) == len(set(test_ids)), 'Repeated test document across folds'
    result['complete'] = len(fold_scores) == expected and not config['partial_experiment']
    if result['complete']:
        assert set(test_ids) == expected_test_ids, 'Missing/extra test documents'
        assert config['epochs'] == 30 and config['limit_train'] == 0 and config['limit_eval'] == 0
    if fold_scores:
        result['mean_fold'] = {mode: {scope: {metric: float(np.mean([
            fold[mode][scope][metric] for fold in fold_scores])) for metric in ['p', 'r', 'f1']}
            for scope in ['all', 'intra', 'inter']} for mode in ['direction', 'existence', 'source_direction']}
        result['pooled'] = scores(all_gold, all_pred, all_types)
        result['f1_from_mean_pr'] = {mode: {scope: (
            2 * values['p'] * values['r'] / (values['p'] + values['r'])
            if values['p'] + values['r'] else 0.) for scope, values in scopes.items()}
            for mode, scopes in result['mean_fold'].items()}
        result['mean_fold_ecg'] = {mode: {scope: {metric: float(np.mean([
            fold[mode][scope][metric] for fold in fold_ecg_scores])) for metric in ['p', 'r', 'f1']}
            for scope in ['all', 'intra', 'inter']} for mode in (
                ['direction', 'existence', 'source_direction'] if result['ecg_direction_metrics_applicable'] else ['existence'])}
        pooled_ecg = scores(all_gold, all_ecg_pred, all_types)
        result['pooled_ecg'] = {mode: pooled_ecg[mode] for mode in result['mean_fold_ecg']}
        result['unique_test_documents'] = len(test_ids)
    return result


def plot_learning(directory, output_dir=None):
    histories = []
    for fold_path in sorted(directory.glob('fold_*')):
        path = fold_path / 'history.jsonl'
        if path.exists():
            history = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if len(history) >= 2:
                histories.append((fold_path.name, history))
    if not histories:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    config = read_json(directory / 'configuration.json')
    selection = selection_metric(config)
    for name, history in histories:
        epochs = [record['epoch'] for record in history]
        axes[0].plot(epochs, [record['train_loss'] for record in history], label=name)
        axes[1].plot(epochs, [record['validation'][selection]['all']['f1'] * 100 for record in history], label=name)
    axes[0].set(ylabel='Training loss', xlabel='Epoch')
    protocol = 'author metric' if selection == 'source_direction' else 'standard metric'
    axes[1].set(ylabel='Development F1 (%s, %%)' % protocol, xlabel='Epoch', ylim=(0, 100))
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(directory.name + ' (observed epochs only)', fontsize=11)
    charts = Path(output_dir) if output_dir is not None else ROOT / 'artifacts/charts'
    charts.mkdir(parents=True, exist_ok=True)
    fig.savefig(charts / (directory.name + '_learning.png'), dpi=180)
    fig.savefig(charts / (directory.name + '_learning.pdf'))
    plt.close(fig)


def main():
    experiments, rows = {}, []
    names = ['esc_direction_full', 'esc_existence_full', 'maven_direction_full']
    names += [path.name for path in sorted((ROOT / 'runs').glob('*_full_incoming'))]
    names += [path.name for path in sorted((ROOT / 'runs').glob('*_full_incoming_paper'))]
    for suffix in dict.fromkeys(['_v1', '_v2', '_v3', PAPER_SUFFIX]):
        names += [path.name for path in sorted((ROOT / 'runs').glob('*_full_incoming_paper' + suffix))]
    for name in names:
        directory = ROOT / 'runs' / name
        if not (directory / 'configuration.json').exists():
            continue
        experiment = inspect_experiment(directory)
        experiments[name] = experiment
        plot_learning(directory)
        if not experiment['complete']:
            continue
        dataset = experiment['configuration']['dataset']
        task = experiment['configuration']['task']
        modes = ['direction', 'source_direction', 'existence'] if task == 'direction' else ['existence']
        for mode in modes:
            # ESC direction-to-binary is diagnostic; the existence paper result
            # requires a separate binary classifier and its own trained weights.
            paper_task = 'direction' if mode == 'source_direction' else mode
            if dataset == 'esc' and task == 'direction' and mode == 'existence':
                continue
            for scope in ['intra', 'inter', 'all']:
                target = PAPER[(dataset, paper_task)][scope]
                measured = experiment['mean_fold'][mode][scope]
                from_pr = experiment['f1_from_mean_pr'][mode][scope] * 100
                pooled = experiment['pooled'][mode][scope]['f1'] * 100
                rows.append(dict(experiment=name, dataset=dataset, trained_task=task,
                    graph_orientation=experiment['configuration'].get('graph_orientation', 'source'),
                    fusion_policy=experiment['configuration'].get('fusion_policy', 'source'),
                    metric_protocol=mode, scope=scope,
                    paper_p=target[0], paper_r=target[1], paper_f1=target[2],
                    measured_p=measured['p'] * 100, measured_r=measured['r'] * 100,
                    measured_f1=measured['f1'] * 100, delta_f1_pp=measured['f1'] * 100 - target[2],
                    measured_f1_from_mean_pr=from_pr, measured_f1_pooled=pooled))
    save_json(ROOT / 'artifacts/verified_experiments.json', experiments)
    markdown = ['# 已复核的 iLIF 实验结果', '',
        '只计入完成全部30轮且通过逐文档预测复核的折。未完成五折时，当前均值不能作为论文五折结果。', '',
        'paper%s 为当前算法：训练状态迭代建图不使用 ω（作者 --rate 默认 0），评估建图仍要求概率≥0.6，最终输出取 argmax。paper_v3 在训练和评估建图都使用 0.6，是本轮对照。paper_v2 的最终输出也按 0.6 过滤，paper_v1 另外没有自环。仅有 paper 后缀的是旧融合修正版，其余为源码对照，不能混合平均。' % PAPER_SUFFIX, '',
        '| 实验 | 图传播版本 | 已复核折数 | 标准 F1 (%) | 作者源码口径 F1 (%) | 论文 F1 (%) |',
        '|---|---|---:|---:|---:|---:|']
    for name, experiment in experiments.items():
        config = experiment['configuration']
        dataset, task = config['dataset'], config['task']
        count = len(experiment['verified_folds'])
        if count:
            standard = experiment['mean_fold'][task]['all']['f1'] * 100
            legacy_mode = 'source_direction' if task == 'direction' else 'existence'
            legacy = experiment['mean_fold'][legacy_mode]['all']['f1'] * 100
            values = ['%.2f' % standard, '%.2f' % legacy]
        else:
            values = ['尚无完整折', '尚无完整折']
        version = ('论文算法 %s（训练建图不用 ω）' % PAPER_SUFFIX[1:] if profile(config) == PAPER_PROFILE else
                   '论文算法 v3（训练建图用 ω，历史）' if profile(config) == PAPER_V3 else
                   '论文算法 v2（阈值输出，历史）' if profile(config) == PAPER_V2 else
                   '论文算法 v1（无自环，历史）' if profile(config) == PAPER_V1 else
                   '仅融合修正（对照）' if config.get('fusion_policy') == 'paper' else
                   '发表前传播＋源码融合（对照）' if config.get('graph_orientation') == 'incoming' else '后续版（对照）')
        markdown.append('| %s | %s | %d/%d | %s | %s | %.1f |' %
            (name, version, count, experiment['expected_folds'], values[0], values[1], PAPER[(dataset, task)]['all'][2]))
        if dataset == 'maven' and task == 'direction' and count:
            value = experiment['mean_fold']['existence']['all']['f1'] * 100
            markdown.append('| MAVEN 存在性（合并方向预测） | %s | 1/1 | %.2f | %.2f | 57.1 |' % (version, value, value))
    markdown += ['', '## 各折详情', '',
        '| 实验 | 折 | 开发集选择的最佳轮次 | 标准 F1 (%) | 源码口径 F1 (%) | 测试正例数 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, experiment in experiments.items():
        task = experiment['configuration']['task']
        legacy = 'source_direction' if task == 'direction' else 'existence'
        for fold in experiment['verified_folds']:
            markdown.append('| %s | %d | %d | %.2f | %.2f | %d |' %
                (name, fold['fold'], fold['best_epoch'], fold['test'][task]['all']['f1'] * 100,
                 fold['test'][legacy]['all']['f1'] * 100, fold['test'][task]['all']['gold']))
    markdown += ['', '## 最终因果图的独立复核', '',
        '源码配置与 paper%s 的主指标来自 argmax，paper_v1/v2 的主指标来自阈值过滤后的最终因果图；右列为同一模型按0.6阈值因果图的诊断值。下列复算不能用于按测试结果切换评价规则。' % PAPER_SUFFIX, '',
        '| 实验 | 配置约定的标准 F1 (%) | 同一模型最终因果图标准 F1 (%) |',
        '|---|---:|---:|']
    for name, experiment in experiments.items():
        if experiment['verified_folds']:
            task = experiment['configuration']['task']
            markdown.append('| %s | %.2f | %.2f |' % (name,
                experiment['mean_fold'][task]['all']['f1'] * 100,
                experiment['mean_fold_ecg'][task]['all']['f1'] * 100))
    markdown += ['', '## 跨折 F1 的汇总方式', '',
        '作者代码提示也可以由平均 P/R 重新计算 F1。三种方式同时列出，不能选择较高者作为预定结果；未完成五折时以下均仅对应已完成折。', '',
        '| 实验 | 各折标准 F1 均值 (%) | 由平均标准 P/R 计算 (%) | 合并全部测试事件对计算 (%) |',
        '|---|---:|---:|---:|']
    for name, experiment in experiments.items():
        if experiment['verified_folds']:
            task = experiment['configuration']['task']
            markdown.append('| %s | %.2f | %.2f | %.2f |' % (name,
                experiment['mean_fold'][task]['all']['f1'] * 100,
                experiment['f1_from_mean_pr'][task]['all'] * 100,
                experiment['pooled'][task]['all']['f1'] * 100))
    markdown += ['', '标准方向指标将方向判错计为 FP 和 FN；作者源码口径漏计这些 FP，因此两列不同。', '',
        '逐文档原始标签、事件对、类别、开发集选择轮次和测试文档隔离已核对。模型重载证据另存为 artifacts/reload_verification_*.json。', '',
        '论文算法修正版状态见 runs/queue_status_paper%s.json；阈值输出与无自环历史版见 runs/queue_status_paper_v2.json、runs/queue_status_paper_v1.json；历史融合修正版见 runs/queue_status_paper.json。完整设置与数据差异见 PAPER_ALGORITHM.md。' % PAPER_SUFFIX]
    (ROOT / 'VERIFIED_RESULTS.md').write_text('\n'.join(markdown) + '\n', encoding='utf-8')
    path = ROOT / 'artifacts/paper_comparison.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        columns = ['experiment', 'dataset', 'trained_task', 'graph_orientation', 'fusion_policy', 'metric_protocol', 'scope', 'paper_p', 'paper_r', 'paper_f1',
                   'measured_p', 'measured_r', 'measured_f1', 'delta_f1_pp',
                   'measured_f1_from_mean_pr', 'measured_f1_pooled']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print({name: dict(complete=result['complete'], verified_folds=len(result['verified_folds']))
           for name, result in experiments.items()})


if __name__ == '__main__':
    main()
