"""Run the two authorized full-model jobs after the frozen ablation queue."""
import argparse
from contextlib import contextmanager
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


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def build_plan(batch):
    original = read(batch / 'plan.json')
    expected = {experiment_name(task, variant) for variant in ABLATIONS for task in TASKS}
    formal = [job for job in original['jobs'] if job['phase'] == 'full']
    if {job['name'] for job in formal} != expected or original['main_experiments_included']:
        raise ValueError('Expected the original six-experiment ablation queue')
    reference = formal[0]['command']
    model = Path(reference[reference.index('--model-path') + 1])
    threads = int(reference[reference.index('--threads') + 1])
    jobs = []
    for task in TASKS:
        command = training_command(sys.executable, ROOT, batch, model, task, 'none', 'full',
                                   original['folds'], original['seed'], threads)
        jobs.append(dict(name=experiment_name(task, 'none'), task=task,
                         variant='none', phase='full', command=command))
    return dict(scope='local_ESC_full_model_tail', main_experiments_included=True,
                predecessor='queue_status.json', required_completed=sorted(expected),
                output_root=str(batch), seed=original['seed'], folds=original['folds'],
                epochs_per_formal_fold=30, jobs=jobs, formal_experiments=2,
                full_fold_models=10, independent_reload_required=True,
                shared_source_sha256=read(batch / 'audit/source_manifest.json'),
                inputs=read(batch / 'audit/preflight.json')['files'],
                tail_source_sha256=sha256(Path(__file__)))


def predecessor_action(state, required):
    if state['status'] == 'complete':
        if not set(required).issubset(state.get('completed', [])):
            raise RuntimeError('Predecessor marked complete without all six ablations')
        return 'ready'
    if state['status'] == 'paused':
        return 'paused'
    if state['status'] in ('failed', 'interrupted', 'blocked'):
        return 'blocked'
    if state['status'] in ('running', 'pausing'):
        return 'waiting_for_ablations'
    raise RuntimeError('Unknown predecessor state: ' + state['status'])


def verify_provenance(plan):
    for name, expected in plan['shared_source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError('Frozen source changed: ' + name)
    if sha256(Path(__file__)) != plan['tail_source_sha256']:
        raise RuntimeError('Full-model tail source changed')
    for name, expected in plan['inputs'].items():
        if sha256(Path(name)) != expected['sha256']:
            raise RuntimeError('Model/data input changed: ' + name)


def disk_check(batch):
    # Six pilots, six ablation groups and two full-model groups; best + last per fold.
    existing = sum(path.stat().st_size for path in batch.rglob('*.pt'))
    estimated_total = (6 + 8 * 5) * 2_000_000_000
    needed = max(0, estimated_total - existing) + 5_000_000_000
    free = shutil.disk_usage(batch).free
    result = dict(status='passed' if free >= needed else 'blocked', free_bytes=free,
                  existing_checkpoint_bytes=existing, estimated_total_bytes=estimated_total,
                  needed_additional_bytes_with_buffer=needed)
    if free < needed:
        raise RuntimeError('Insufficient disk space for ablations plus full-model tail: ' + str(result))
    return result


@contextmanager
def exclusive_lock(path):
    import msvcrt
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('The full-model tail is already running')
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def assert_no_batch_worker(batch):
    query = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*" + batch.name + "*' -and "
             "($_.CommandLine -like '*reproduction\\run.py*' -or "
             "$_.CommandLine -like '*reproduction\\infer.py*') } | "
             'Select-Object -ExpandProperty ProcessId')
    live = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command', query], text=True).strip()
    if live:
        raise RuntimeError('A training/inference worker is still active: ' + live)


def execute(batch, plan):
    status_path = batch / 'main_model_tail_status.json'
    previous = read(status_path) if status_path.exists() else {}
    status = dict(pid=os.getpid(), status='waiting_for_ablations',
                  stage='waiting_for_ablations', main_experiments_included=True,
                  started=time.strftime('%Y-%m-%d %H:%M:%S'), completed=previous.get('completed', []),
                  jobs=[job['name'] for job in plan['jobs']])

    def update(**values):
        status.update(values, updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        save(status_path, status)

    def run(command, log_path, stage, label):
        update(status='running', stage=stage, current=label)
        print(stage, label, flush=True)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write('\n' + json.dumps(command, ensure_ascii=False) + '\n')
            stream.flush()
            child = subprocess.Popen(command, cwd=str(ROOT), env=os.environ.copy(),
                                     stdout=stream, stderr=subprocess.STDOUT)
            update(child_pid=child.pid)
            code = child.wait()
            status.pop('child_pid', None)
            if code:
                raise RuntimeError('Exit %d for %s; see %s' % (code, label, log_path))

    update()
    try:
        while True:
            predecessor = read(batch / plan['predecessor'])
            action = predecessor_action(predecessor, plan['required_completed'])
            update(status=action, stage=action, predecessor_current=predecessor.get('current'))
            if action in ('paused', 'blocked'):
                print('Full-model tail stopped: predecessor is ' + predecessor['status'], flush=True)
                return
            if action == 'ready':
                break
            time.sleep(5)
        assert_no_batch_worker(batch)
        verify_provenance(plan)
        save(batch / 'audit/main_model_tail_disk_check.json', disk_check(batch))
        # Recompute the predecessor gates instead of trusting only its status label.
        from analyze_results import inspect_experiment
        from analyze_ablations import verify_reloads
        for name in plan['required_completed']:
            directory = batch / name
            checked = inspect_experiment(directory)
            if not checked['complete'] or len(verify_reloads(directory, checked['verified_folds'])) != 5:
                raise RuntimeError('Ablation evidence is incomplete: ' + name)
        for job in plan['jobs']:
            verify_provenance(plan)
            assert_no_batch_worker(batch)
            directory = batch / job['name']
            log_path = batch / (job['name'] + '.log')
            run(job['command'], log_path, 'training', job['name'])
            for fold in plan['folds']:
                checkpoint = directory / ('fold_%d/best.pt' % fold)
                predictions = checkpoint.parent / 'reloaded_predictions.json'
                label = job['name'] + '/fold_%d' % fold
                run([sys.executable, '-u', str(ROOT / 'infer.py'), '--checkpoint', str(checkpoint),
                     '--output', str(predictions)], log_path, 'reload_inference', label)
                run([sys.executable, '-u', str(ROOT / 'verify_reload.py'), '--checkpoint', str(checkpoint),
                     '--prediction-file', str(predictions)], log_path, 'reload_verification', label)
            checked = inspect_experiment(directory)
            if not checked['complete'] or len(verify_reloads(directory, checked['verified_folds'])) != 5:
                raise RuntimeError('Full-model evidence is incomplete: ' + job['name'])
            if job['name'] not in status['completed']:
                status['completed'].append(job['name'])
            update()
            run([sys.executable, '-u', str(ROOT / 'analyze_ablations.py'), '--runs-root', str(batch),
                 '--output-dir', str(batch / 'reports'), '--variants', 'none'] + list(ABLATIONS),
                batch / 'analysis.full_model_tail.log', 'reporting', job['name'])
        update(status='complete', stage='complete', finished=time.strftime('%Y-%m-%d %H:%M:%S'))
        print('COMPLETE: both full-model tasks trained and independently reloaded', flush=True)
    except BaseException as error:
        update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, default=ROOT / 'runs/esc_local_ablations_v4_20261105')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
    batch = cli.batch_root.resolve()
    plan = build_plan(batch)
    if cli.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    with exclusive_lock(batch / '.main_model_tail.lock'):
        plan_path = batch / 'main_model_tail_plan.json'
        if plan_path.exists() and read(plan_path) != plan:
            raise RuntimeError('The authorized full-model tail plan or its inputs changed')
        verify_provenance(plan)
        save(batch / 'audit/main_model_tail_disk_check.json', disk_check(batch))
        save(plan_path, plan)
        original = read(batch / 'plan.json')
        combined = dict(original, scope='local_ESC_ablations_then_full_model',
                        main_experiments_included=True, jobs=original['jobs'] + plan['jobs'],
                        formal_experiments=8, full_fold_models=40,
                        status_files=['queue_status.json', 'main_model_tail_status.json'])
        save(batch / 'plan_with_full_model.json', combined)
        snapshot = batch / 'audit/main_model_tail_source/queue_full_model_tail.py'
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), snapshot)
        if cli.prepare_only:
            print('Prepared: 12 existing jobs followed by two full-model jobs; no training started')
            return
        os.environ.update(PYTHONHASHSEED=str(plan['seed']), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        execute(batch, plan)


if __name__ == '__main__':
    main()
