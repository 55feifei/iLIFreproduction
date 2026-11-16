"""Local v4 ablations only: checks, six pilots, six five-fold runs, reloads."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from ablation_config import TASKS, experiment_name, training_command
from ablation_preflight import sha256

ROOT = Path(__file__).resolve().parent
ABLATIONS = ('no_direction', 'no_heterogeneity', 'no_iteration')
SOURCE_FILES = ('run.py', 'run_ablations.py', 'local_ablation_queue.py', 'common.py',
    'project_paths.py', 'algorithm_config.py', 'fusion_config.py', 'paper_model.py',
    'adapted_model.py', 'model/model.py', 'model/CGE.py', 'legacy_utils.py', 'infer.py',
    'verify_reload.py', 'analyze_results.py', 'analyze_ablations.py', 'ablation_config.py',
    'ablation_preflight.py', 'check_ablations.py', 'check_paper_algorithm.py',
    'check_paper_fusion.py', 'check_local_ablation_queue.py')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'runs/esc_local_ablations_v4_20261105')
    parser.add_argument('--model-path', type=Path, default=ROOT/'models/bert-base-uncased')
    parser.add_argument('--variants', choices=ABLATIONS, nargs='+', default=list(ABLATIONS))
    parser.add_argument('--tasks', choices=TASKS, nargs='+', default=list(TASKS))
    parser.add_argument('--graph-orientation', choices=['incoming'], default='incoming')
    parser.add_argument('--seed', type=int, default=209)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--pilots-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    cli = parser.parse_args(argv)
    if cli.threads < 1:
        parser.error('--threads must be positive')
    for field in ('variants', 'tasks'):
        if len(getattr(cli, field)) != len(set(getattr(cli, field))):
            parser.error('Duplicate --'+field)
    cli.output_root, cli.model_path = cli.output_root.resolve(), cli.model_path.resolve()
    if cli.output_root == (ROOT/'runs').resolve():
        parser.error('Use a separate batch directory to preserve historical runs')
    return cli


def build_plan(cli):
    jobs = []
    for phase in (['pilot'] if cli.pilots_only else ['pilot', 'full']):
        for variant in cli.variants:
            for task in cli.tasks:
                assert variant in ABLATIONS, 'Main model training is excluded'
                command = training_command(sys.executable, ROOT, cli.output_root,
                    cli.model_path, task, variant, phase, (1,2,3,4,5), cli.seed, cli.threads)
                jobs.append(dict(name=experiment_name(task,variant,phase), phase=phase,
                                 variant=variant, task=task, command=command))
    return dict(scope='local_ESC_ablation_only', main_experiments_included=False,
        algorithm_profile='paper_algorithm_v4', seed=cli.seed, folds=[1,2,3,4,5], jobs=jobs,
        formal_experiments=0 if cli.pilots_only else len(cli.variants)*len(cli.tasks),
        full_fold_models=0 if cli.pilots_only else len(cli.variants)*len(cli.tasks)*5,
        epochs_per_formal_fold=30, independent_reload_required=True,
        fold_rng_policy='continuous_across_folds_as_historical_local_v4',
        data_policy='original_local_author_cache_same_as_existing_local_v4',
        output_root=str(cli.output_root))


def process_alive(pid):
    if os.name == 'nt':
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value==259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid),0)
        return True
    except OSError:
        return False


def main(argv=None):
    cli = arguments(argv)
    plan = build_plan(cli)
    if cli.dry_run:
        print(json.dumps(plan,indent=2,ensure_ascii=False))
        return
    from common import ESC_DATA
    from ablation_preflight import check
    os.environ.update(PYTHONHASHSEED=str(cli.seed), CUBLAS_WORKSPACE_CONFIG=':4096:8',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    cli.output_root.mkdir(parents=True,exist_ok=True)
    status_path = cli.output_root/'queue_status.json'
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding='utf-8'))
        if old.get('status')=='running' and process_alive(old['pid']):
            raise RuntimeError('This batch is already running: PID %s'%old['pid'])
    preflight = check(cli.model_path, ESC_DATA, cli.tasks)
    preflight['output_root'] = str(cli.output_root)
    preflight['free_disk_bytes'] = shutil.disk_usage(cli.output_root).free
    retained_models = len(plan['jobs'])//2*6 if not cli.pilots_only else len(plan['jobs'])
    preflight['estimated_checkpoint_bytes'] = retained_models*2_000_000_000
    existing_bytes = sum(path.stat().st_size for path in cli.output_root.rglob('*.pt'))
    needed_bytes = max(0,preflight['estimated_checkpoint_bytes']-existing_bytes)+5_000_000_000
    if preflight['free_disk_bytes'] < needed_bytes:
        preflight['errors'].append('Insufficient free disk for checkpoints: need %d bytes'%needed_bytes)
    preflight['status'] = 'passed' if not preflight['errors'] else 'blocked'
    save_json(cli.output_root/'audit/preflight.json',preflight)
    save_json(cli.output_root/'plan.json',plan)
    print(json.dumps(preflight,ensure_ascii=False),flush=True)
    if preflight['errors']:
        raise RuntimeError('Preflight failed: '+'; '.join(preflight['errors']))
    if cli.preflight_only:
        return
    hashes = {name:sha256(ROOT/name) for name in SOURCE_FILES}
    manifest_path = cli.output_root/'audit/source_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8'))!=hashes:
            raise RuntimeError('Batch code changed; use archived code or a new batch directory')
    else:
        save_json(manifest_path,hashes)
        for name in SOURCE_FILES:
            destination = cli.output_root/'audit/source'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,destination)
    status = dict(pid=os.getpid(),status='running',scope=plan['scope'],main_experiments_included=False,
        started=time.strftime('%Y-%m-%d %H:%M:%S'),completed=[],plan=plan)

    def run(command,log_path,stage,label):
        status.update(stage=stage,current=label)
        save_json(status_path,status)
        print('%s %s'%(stage,label),flush=True)
        with log_path.open('a',encoding='utf-8') as stream:
            stream.write('\n'+json.dumps(command,ensure_ascii=False)+'\n');stream.flush()
            child = subprocess.Popen(command,cwd=str(ROOT),env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT)
            status['child_pid'] = child.pid
            save_json(status_path,status)
            code = child.wait()
            status.pop('child_pid',None)
            if code:
                raise RuntimeError('Exit %d for %s; see %s'%(code,label,log_path))

    try:
        for filename in ('check_local_ablation_queue.py','check_paper_algorithm.py',
                         'check_paper_fusion.py','check_ablations.py'):
            run([sys.executable,'-u',str(ROOT/filename)],cli.output_root/'audit/checks.log','semantic_checks',filename)
        for name in ('paper_algorithm_checks.json','paper_fusion_checks.json','ablation_checks.json'):
            shutil.copy2(ROOT/'artifacts'/name,cli.output_root/'audit'/name)
        from analyze_results import inspect_experiment
        from analyze_ablations import verify_reloads
        for job in plan['jobs']:
            assert job['variant'] in ABLATIONS,'Refuse to train the main model'
            if hashes!={name:sha256(ROOT/name) for name in SOURCE_FILES}:
                raise RuntimeError('Source changed during batch; refusing to mix implementations')
            directory = cli.output_root/job['name']
            log_path = cli.output_root/(job['name']+'.log')
            run(job['command'],log_path,'training',job['name'])
            for fold in ([1] if job['phase']=='pilot' else plan['folds']):
                checkpoint = directory/('fold_%d/best.pt'%fold)
                predictions = checkpoint.parent/'reloaded_predictions.json'
                run([sys.executable,'-u',str(ROOT/'infer.py'),'--checkpoint',str(checkpoint),
                    '--output',str(predictions)],log_path,'reload_inference',job['name']+'/fold_%d'%fold)
                run([sys.executable,'-u',str(ROOT/'verify_reload.py'),'--checkpoint',str(checkpoint),
                    '--prediction-file',str(predictions)],log_path,'reload_verification',job['name']+'/fold_%d'%fold)
            checked = inspect_experiment(directory)
            reloaded = verify_reloads(directory,checked['verified_folds'])
            expected = 1 if job['phase']=='pilot' else 5
            assert len(checked['verified_folds'])==len(reloaded)==expected,'Verification incomplete'
            if job['phase']=='full':
                assert checked['complete']
            status['completed'].append(job['name'])
            save_json(status_path,status)
            if job['phase']=='full':
                run([sys.executable,'-u',str(ROOT/'analyze_ablations.py'),'--runs-root',str(cli.output_root),
                    '--output-dir',str(cli.output_root/'reports'),'--variants']+list(cli.variants),
                    cli.output_root/'analysis.log','reporting',job['name'])
        status.update(status='complete',stage='complete',finished=time.strftime('%Y-%m-%d %H:%M:%S'))
        save_json(status_path,status)
        print('COMPLETE: ablations only; all formal folds independently reloaded and verified',flush=True)
    except BaseException as exc:
        status.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                      error=str(exc),finished=time.strftime('%Y-%m-%d %H:%M:%S'))
        save_json(status_path,status)
        raise


if __name__=='__main__':
    main()
