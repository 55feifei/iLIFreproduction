"""Recompute Table 3, including the matched full-model baseline and reload gates."""
import argparse
import csv
import json
from pathlib import Path
from common import ROOT, save_json
from analyze_results import inspect_experiment, plot_learning
from algorithm_config import PAPER
from ablation_config import TARGETS, VARIANTS, TASKS, experiment_name, PAPER_URL
from ablation_preflight import sha256


def verify_reloads(directory, folds):
    verified = []
    config_hash = sha256(directory / 'configuration.json')
    for fold in folds:
        fold_path = directory / ('fold_%d' % fold['fold'])
        path = fold_path / 'reload_verification.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding='utf-8'))
        assert report['metrics_match'] and report['causal_graphs_match'], path
        assert report['configuration_sha256'] == config_hash, ('Stale reload configuration', path)
        assert report['matched_documents'] == len(json.loads((fold_path / 'best_predictions.json').read_text(encoding='utf-8')))
        for filename in ['best.pt', 'best_predictions.json', 'result.json', 'split.json']:
            assert report['sha256'][filename] == sha256(fold_path / filename), ('Stale reload', path, filename)
        prediction_file = fold_path / 'reloaded_predictions.json'
        assert report['prediction_sha256'] == sha256(prediction_file), ('Reloaded predictions changed', path)
        verified.append(fold['fold'])
    return verified


def validate_comparison(baseline, variant, baseline_dir, variant_dir):
    # Compare only experiments trained on the same inputs and model contract.
    for key in ['algorithm_profile', 'task', 'seed', 'epochs', 'data_sha256', 'model_path',
                'model_weights_sha256', 'checkpoint_bert', 'numeric_precision']:
        assert baseline['configuration'].get(key) == variant['configuration'].get(key), ('Unmatched baseline', key)
    for fold in variant['verified_folds']:
        name = 'fold_%d/split.json' % fold['fold']
        assert json.loads((baseline_dir / name).read_text(encoding='utf-8')) == json.loads(
            (variant_dir / name).read_text(encoding='utf-8')), ('Unmatched fold split', name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'runs')
    parser.add_argument('--output-dir', type=Path,
                        help='Keep this batch report and evidence separate from historical reports')
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    cli = parser.parse_args(argv)
    report_root = cli.output_dir
    if report_root is not None:
        report_root.mkdir(parents=True, exist_ok=True)
    artifact_root = report_root if report_root is not None else ROOT / 'artifacts'
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiments, rows = {}, []
    for variant in cli.variants:
        for task in TASKS:
            name = experiment_name(task, variant)
            directory = cli.runs_root / name
            if not (directory / 'configuration.json').exists():
                continue
            config = json.loads((directory / 'configuration.json').read_text(encoding='utf-8'))
            assert config.get('algorithm_profile') == PAPER and config.get('ablation', 'none') == variant
            assert config['dataset'] == 'esc' and config['task'] == task
            result = inspect_experiment(directory)
            if report_root is not None:
                plot_learning(directory, report_root / 'charts')
            else:
                plot_learning(directory)
            result['prediction_artifacts_complete'] = result['complete']
            result['reloaded_folds'] = verify_reloads(directory, result['verified_folds'])
            result['complete'] = result['complete'] and len(result['reloaded_folds']) == 5
            experiments[name] = result
    lines = ['# 核心消融实验（论文表 3）', '',
        '论文：[第 4.4 节、表 3](%s)。当前算法：`%s`。' % (PAPER_URL, PAPER), '',
        '各变体独立训练，方向和存在性分别训练，固定开发集，五折×30轮。',
        '实测列为逐折测试集标准 P/R/F1 的算术平均；ΔF1 为变体减去同配置完整模型，单位为百分点。',
        '下表未完成行仅表示进度；只有五折训练、预测复算及独立模型重载均通过时才写入正式 CSV。', '',
        '| 变体 | 任务 | 训练复核/重载折数 | 实测 P/R/F1 (%) | ΔF1 | 论文 P/R/F1 (%) |',
        '|---|---|---:|---|---:|---|']
    if 'none' not in cli.variants:
        lines += ['本批仅训练消融变体，没有重新训练完整模型；无同批完整模型时不计算 ΔF1。', '']
    for variant in cli.variants:
        for task in TASKS:
            name = experiment_name(task, variant)
            result = experiments.get(name)
            baseline_name = experiment_name(task, 'none')
            baseline = experiments.get(baseline_name)
            count = len(result['verified_folds']) if result else 0
            reloaded = len(result['reloaded_folds']) if result else 0
            values = result['mean_fold'][task]['all'] if count else None
            delta = None
            if result and result['complete'] and baseline and baseline['complete']:
                validate_comparison(baseline, result, cli.runs_root / baseline_name, cli.runs_root / name)
                delta = (values['f1'] - baseline['mean_fold'][task]['all']['f1']) * 100
            target = TARGETS[variant][task]
            observed = '/'.join('%.2f' % (values[m] * 100) for m in ('p', 'r', 'f1')) if values else '待运行'
            if count and not result['complete']:
                observed += '（未完成核验）'
            lines.append('| %s | %s | %d/5 · %d/5 | %s | %s | %s |' % (
                variant, task, count, reloaded, observed, '%+.2f' % delta if delta is not None else '—',
                '/'.join('%.1f' % value for value in target)))
            if result and result['complete']:
                rows.append(dict(variant=variant, task=task, metric_protocol=task,
                    paper_p=target[0], paper_r=target[1], paper_f1=target[2],
                    measured_p=values['p']*100, measured_r=values['r']*100, measured_f1=values['f1']*100,
                    delta_f1_pp=delta, delta_paper_f1_pp=values['f1']*100-target[2],
                    f1_from_mean_pr=result['f1_from_mean_pr'][task]['all']*100))
    lines += ['', '`no_direction`：因果边双向化，分类头仍保留原任务标签，邻接差基于无向图。',
        '`no_heterogeneity`：所有因果边及自环使用同一关系类型/注意力分支，不使用句内与跨句加权融合。',
        '`no_iteration`：初始预测建图后只进行一次图编码，再分类；仅最终预测计入损失，图编码器参与训练。',
        '其余设置统一沿用项目当前 v4；训练 argmax 建图，评估阈值0.6，最终 argmax 输出，开发集标准 F1 选轮。',
        '论文未说明的实现取法和数据限制见 PAPER_ALGORITHM.md；不能据此保证逐位重现作者数值。']
    markdown_path = report_root / 'ABLATION_RESULTS.md' if report_root is not None else ROOT / 'ABLATION_RESULTS.md'
    with markdown_path.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines) + '\n')
    save_json(artifact_root / 'verified_ablations.json', experiments)
    with (artifact_root / 'ablation_comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, lineterminator='\n', fieldnames=['variant','task','metric_protocol','paper_p','paper_r','paper_f1',
            'measured_p','measured_r','measured_f1','delta_f1_pp','delta_paper_f1_pp','f1_from_mean_pr'])
        writer.writeheader()
        writer.writerows(rows)
    print({name: dict(trained=len(result['verified_folds']), reloaded=len(result['reloaded_folds']),
                     complete=result['complete']) for name, result in experiments.items()})


if __name__ == '__main__':
    main()
