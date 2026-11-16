"""Verify all primary experiments after the training queue actually exits.

No training, ablations, tuning, or test-set model selection is performed here.
The optional process wait uses a real process handle, never a status-file timer.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import ROOT, save_json
from fusion_config import fusion_from_config, fusion_settings
from algorithm_config import PAPER, TRAINABLE_PROFILES, PAPER_SUFFIX, profile, validate_contract


PRIMARY = ['esc_direction_full_incoming', 'esc_existence_full_incoming',
           'maven_direction_full_incoming']
STATUS = ROOT / 'artifacts/main_verification_status.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(script, arguments, environment, log):
    command = [sys.executable, '-u', str(ROOT / script), *map(str, arguments)]
    print('Verify:', script, *map(str, arguments), flush=True)
    subprocess.run(command, cwd=str(ROOT), env=environment, stdout=log,
                   stderr=subprocess.STDOUT, check=True)


def render_primary(experiments, primary=None, fusion_policy='paper', algorithm_profile=PAPER):
    if primary is None:
        suffix = '_paper' + PAPER_SUFFIX if algorithm_profile == PAPER else ('_paper' if fusion_policy == 'paper' else '')
        primary = [name + suffix for name in PRIMARY]
    from analyze_results import PAPER
    entries = [
        ('EventStoryLine 方向', primary[0], 'esc', 'direction'),
        ('EventStoryLine 存在性', primary[1], 'esc', 'existence'),
        ('MAVEN-ERE 方向', primary[2], 'maven', 'direction'),
        ('MAVEN-ERE 存在性', primary[2], 'maven', 'existence'),
    ]
    lines = ['# 论文主表复现结果', '',
             '仅包含本次确认的主表实验。全部训练、测试覆盖检查和独立模型重载检查已经完成。', '',
             '单位为百分比。EventStoryLine 为五折 F1 的算术平均；MAVEN-ERE 为完整测试集微平均。', '',
             '| 实验 | 范围 | 论文 P / R / F1 | 实测标准 P / R / F1 | 源码口径 F1 | 标准 F1 差值（百分点） |',
             '|---|---|---|---|---:|---:|']
    values = []
    for label, name, dataset, task in entries:
        experiment = experiments[name]
        for scope, scope_label in [('intra', '句内'), ('inter', '跨句'), ('all', '总体')]:
            paper = PAPER[(dataset, task)][scope]
            measured = experiment['mean_fold'][task][scope]
            source_mode = 'source_direction' if task == 'direction' else 'existence'
            source_f1 = experiment['mean_fold'][source_mode][scope]['f1'] * 100
            measured_values = [measured[key] * 100 for key in ['p', 'r', 'f1']]
            delta = measured_values[2] - paper[2]
            lines.append('| %s | %s | %s | %s | %.2f | %+.2f |' % (
                label, scope_label, ' / '.join('%.1f' % v for v in paper),
                ' / '.join('%.2f' % v for v in measured_values), source_f1, delta))
            values.append(dict(experiment=name, dataset=dataset, task=task, scope=scope,
                               paper=dict(zip(['p', 'r', 'f1'], paper)),
                               standard=dict(zip(['p', 'r', 'f1'], measured_values)),
                               source_f1=source_f1, delta_f1_pp=delta))
    lines += ['',
        ('算法版本 %s：类型融合句内0.7、跨句0.3，句内路径为每个节点加自环（论文图2）；完整算法约定见 PAPER_ALGORITHM.md。' % PAPER
         if fusion_policy == 'paper' else '此表为作者示例命令的0.3/0.7及位置融合对照，不能称为论文公式（4）设置。'), '',
        ('主表按最后一轮argmax评价（0.6阈值只用于迭代建图），开发集使用同一标准指标选轮。方向判错同时计为 FP 和 FN；源码口径仅作诊断。'
         if algorithm_profile == PAPER else '历史源码配置使用 argmax 及原开发集选轮方式；本表为对照。'), '',
        '训练采用发表前官方快照的传播方向。版本、环境和数据协议差异见 PUBLICATION_VERSION.md、README.md 和 RESULTS.md。数值差距如实保留，不能将流程完成等同于达到论文数值。', '',
        '每个模型的重载证据见 artifacts/reload_verification_*_incoming_fold_*.json；逐折指标、选轮和三种跨折汇总见 VERIFIED_RESULTS.md；机器可读对照见 artifacts/primary_table_results.json。', '']
    title = 'PRIMARY_TABLE_RESULTS' if algorithm_profile == PAPER else 'SOURCE_COMMAND_RESULTS'
    (ROOT / (title + '.md')).write_text('\n'.join(lines), encoding='utf-8')
    save_json(ROOT / ('artifacts/' + title.lower() + '.json'), values)


def main():
    global STATUS
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--fusion-policy', choices=['source', 'paper'], default='paper')
    parser.add_argument('--algorithm-profile', choices=TRAINABLE_PROFILES, default=PAPER)
    cli = parser.parse_args()
    suffix = '_paper' if cli.fusion_policy == 'paper' else ''
    if cli.algorithm_profile == PAPER:
        if cli.fusion_policy != 'paper':
            parser.error('Paper algorithm requires paper fusion')
        suffix += PAPER_SUFFIX
    primary = [name + suffix for name in PRIMARY]
    STATUS = ROOT / ('artifacts/main_verification_status%s.json' % suffix)
    environment = os.environ.copy()
    environment.update(PYTHONHASHSEED='209', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    state = dict(pid=os.getpid(), python=sys.executable, scope='primary_tables_only',
                 started=time.strftime('%Y-%m-%d %H:%M:%S'), verified_models=[])
    state.update(fusion_settings(cli.fusion_policy))
    state.update(algorithm_profile=cli.algorithm_profile)
    try:
        if cli.wait_pid:
            import psutil
            process = psutil.Process(cli.wait_pid)
            arguments = process.cmdline()
            script_arguments = [argument for argument in arguments[1:] if argument.lower().endswith('.py')]
            assert len(script_arguments) == 1, arguments
            driver = Path(script_arguments[0])
            if not driver.is_absolute():
                driver = Path(process.cwd()) / driver
            assert driver.resolve() == (ROOT / 'run_publication.py').resolve(), arguments
            state.update(status='waiting_for_training_process', training_pid=process.pid,
                         training_process_created=process.create_time(), training_command=arguments)
            save_json(STATUS, state)
            print('Waiting for the verified primary training process:', process.pid, flush=True)
            while True:
                try:
                    process.wait(timeout=15)
                    break
                except psutil.TimeoutExpired:
                    pass
        queue = read_json(ROOT / ('runs/queue_status%s.json' % suffix))
        assert queue['status'] == 'complete', ('Training queue did not finish', queue)
        assert set(primary).issubset(queue['completed'])
        assert queue['graph_implementation'] == 'official_pre_publication'
        assert fusion_from_config(queue) == fusion_settings(cli.fusion_policy)
        assert profile(queue) == cli.algorithm_profile
        state.update(status='verifying', training_finished=queue.get('finished'))
        save_json(STATUS, state)
        with (ROOT / ('artifacts/main_verification%s.log' % suffix)).open('a', encoding='utf-8') as log:
            run('analyze_results.py', [], environment, log)
            experiments = read_json(ROOT / 'artifacts/verified_experiments.json')
            for name in primary:
                experiment = experiments[name]
                assert experiment['complete'], name
                assert experiment['configuration'].get('ablation', 'none') == 'none'
                assert experiment['configuration']['graph_orientation'] == 'incoming'
                assert fusion_from_config(experiment['configuration']) == fusion_settings(cli.fusion_policy)
                assert profile(experiment['configuration']) == cli.algorithm_profile
                validate_contract(experiment['configuration'])
                for fold in experiment['verified_folds']:
                    directory = ROOT / 'runs' / name / ('fold_%d' % fold['fold'])
                    checkpoint = directory / 'best.pt'
                    predictions = ROOT / 'artifacts' / ('reloaded_%s_fold_%d.json' % (name, fold['fold']))
                    state.update(current_experiment=name, current_fold=fold['fold'])
                    save_json(STATUS, state)
                    run('infer.py', ['--checkpoint', checkpoint, '--output', predictions], environment, log)
                    run('verify_reload.py', ['--checkpoint', checkpoint, '--prediction-file', predictions], environment, log)
                    evidence_path = ROOT / 'artifacts' / ('reload_verification_%s_fold_%d.json' % (name, fold['fold']))
                    evidence = read_json(evidence_path)
                    assert evidence['metrics_match'] and evidence['causal_graphs_match']
                    assert evidence['matched_documents'] == len(read_json(directory / 'split.json')['test'])
                    state['verified_models'].append(dict(experiment=name, fold=fold['fold'], evidence=str(evidence_path)))
                    save_json(STATUS, state)
            assert len(state['verified_models']) == 11
            run('analyze_results.py', [], environment, log)
            run('report.py', [], environment, log)
        render_primary(read_json(ROOT / 'artifacts/verified_experiments.json'), primary, cli.fusion_policy, cli.algorithm_profile)
        state.update(status='complete', finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                     models=11, esc_direction_folds=5, esc_existence_folds=5,
                     maven_direction_and_existence=True)
        save_json(STATUS, state)
        print('Primary training and all 11 model reload verifications completed.', flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                     failed=time.strftime('%Y-%m-%d %H:%M:%S'))
        save_json(STATUS, state)
        raise


if __name__ == '__main__':
    main()
