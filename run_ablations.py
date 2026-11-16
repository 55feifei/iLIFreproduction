"""Serial Table 3 queue: checks -> pilots -> full training -> independent reload."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from project_paths import ROOT, esc_data_path
from algorithm_config import PAPER, PAPER_SUFFIX
from ablation_config import VARIANTS, TASKS, experiment_name, training_command


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilots-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print JSON plan without dependencies or side effects')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    parser.add_argument('--tasks', choices=TASKS, nargs='+', default=list(TASKS))
    parser.add_argument('--folds', choices=range(1, 6), type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--graph-orientation', choices=['incoming'], default='incoming')
    parser.add_argument('--model-path', type=Path, default=ROOT / 'models/bert-base-uncased')
    parser.add_argument('--esc-data', type=Path, default=esc_data_path())
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    parser.add_argument('--seed', type=int, default=209)
    parser.add_argument('--threads', type=int, default=4)
    cli = parser.parse_args(argv)
    for field in ('variants', 'tasks', 'folds'):
        if len(getattr(cli, field)) != len(set(getattr(cli, field))):
            parser.error('Duplicate --' + field)
    if cli.threads < 1:
        parser.error('--threads must be positive')
    cli.model_path, cli.esc_data, cli.output_root = [path.expanduser().resolve()
        for path in (cli.model_path, cli.esc_data, cli.output_root)]
    return cli


def build_plan(cli):
    jobs = []
    for phase in (['pilot'] if cli.pilots_only else ['pilot', 'full']):
        for variant in cli.variants:
            for task in cli.tasks:
                jobs.append(dict(name=experiment_name(task, variant, phase), phase=phase,
                    task=task, variant=variant,
                    command=training_command(sys.executable, ROOT, cli.output_root,
                        cli.model_path, task, variant, phase, cli.folds, cli.seed, cli.threads)))
    return dict(algorithm_profile=PAPER, esc_data=str(cli.esc_data),
        seed=cli.seed, folds=cli.folds, jobs=jobs,
        full_fold_models=0 if cli.pilots_only else len(cli.variants) * len(cli.tasks) * len(cli.folds),
        protocol='independent training; 30 epochs; dev standard F1; mean test F1 across five folds',
        reload_verification=True)


def run_logged(command, log_path, environment):
    print('Running %s; log: %s' % (Path(command[2]).name, log_path), flush=True)
    with log_path.open('a', encoding='utf-8') as stream:
        stream.write('\n' + json.dumps(command, ensure_ascii=False) + '\n')
        stream.flush()
        result = subprocess.run(command, cwd=str(ROOT), env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Command failed (%d); see %s' % (result.returncode, log_path))


def main(argv=None):
    cli = parse_args(argv)
    plan = build_plan(cli)
    if cli.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    environment = os.environ.copy()
    environment.update(PYTHONHASHSEED=str(cli.seed), CUBLAS_WORKSPACE_CONFIG=':4096:8',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
        ILIF_ESC_DATA=str(cli.esc_data))
    os.environ['ILIF_ESC_DATA'] = str(cli.esc_data)
    from ablation_preflight import check
    preflight = check(cli.model_path, cli.esc_data, cli.tasks)
    save_json(ROOT / 'artifacts/ablation_preflight.json', preflight)
    print(json.dumps(preflight, indent=2, ensure_ascii=False), flush=True)
    if preflight['errors']:
        raise SystemExit('Preflight failed; fix the listed dependencies and input paths')
    if cli.preflight_only:
        return
    cli.output_root.mkdir(parents=True, exist_ok=True)
    status_path = cli.output_root / ('ablation_queue_status_paper%s.json' % PAPER_SUFFIX)
    save_json(cli.output_root / ('ablation_plan_paper%s.json' % PAPER_SUFFIX), plan)
    status = dict(pid=os.getpid(), status='running', algorithm_profile=PAPER,
                  completed=[], plan=plan, preflight=preflight)
    try:
        status['current'] = 'semantic_checks'
        save_json(status_path, status)
        run_logged([sys.executable, '-u', str(ROOT / 'check_ablation_queue.py')],
                   cli.output_root / 'ablation_checks.log', environment)
        run_logged([sys.executable, '-u', str(ROOT / 'check_ablations.py')],
                   cli.output_root / 'ablation_checks.log', environment)
        for job in plan['jobs']:
            name = job['name']
            status.update(current=name, stage='training', started=time.strftime('%Y-%m-%d %H:%M:%S'))
            save_json(status_path, status)
            log_path = cli.output_root / (name + '.log')
            run_logged(job['command'], log_path, environment)
            status['stage'] = 'reload_verification'
            save_json(status_path, status)
            directory = cli.output_root / name
            for fold in ([1] if job['phase'] == 'pilot' else cli.folds):
                checkpoint = directory / ('fold_%d/best.pt' % fold)
                predictions = checkpoint.parent / 'reloaded_predictions.json'
                run_logged([sys.executable, '-u', str(ROOT / 'infer.py'), '--checkpoint',
                    str(checkpoint), '--output', str(predictions)], log_path, environment)
                run_logged([sys.executable, '-u', str(ROOT / 'verify_reload.py'), '--checkpoint',
                    str(checkpoint), '--prediction-file', str(predictions)], log_path, environment)
            status['completed'].append(name)
            save_json(status_path, status)
            if job['phase'] == 'full':
                run_logged([sys.executable, '-u', str(ROOT / 'analyze_ablations.py'),
                    '--runs-root', str(cli.output_root),
                    '--output-dir', str(cli.output_root / 'reports'),
                    '--variants'] + list(cli.variants), log_path, environment)
        status.update(status='complete', stage='complete', finished=time.strftime('%Y-%m-%d %H:%M:%S'))
        save_json(status_path, status)
    except BaseException as exc:
        status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        save_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
