"""Sequential GPU queue with logs and failure visibility."""
import os
import subprocess
import sys
import time
from common import ROOT, save_json
from report import main as report


def main():
    queue = [
        ('esc_direction_pilot', ['--dataset', 'esc', '--task', 'direction', '--epochs', '1', '--folds', '1', '--limit-train', '8', '--limit-eval', '4']),
        ('esc_existence_pilot', ['--dataset', 'esc', '--task', 'existence', '--epochs', '1', '--folds', '1', '--limit-train', '8', '--limit-eval', '4']),
        ('maven_direction_pilot_vocab30524', ['--dataset', 'maven', '--epochs', '1', '--limit-train', '8', '--limit-eval', '4']),
        ('esc_direction_full', ['--dataset', 'esc', '--task', 'direction']),
        ('esc_existence_full', ['--dataset', 'esc', '--task', 'existence']),
        ('maven_direction_full', ['--dataset', 'maven']),
    ]
    environment = os.environ.copy()
    environment.update(PYTHONHASHSEED='209', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    status = dict(pid=os.getpid(), python=sys.executable, status='running', completed=[])
    for name, args in queue:
        status.update(current=name, started=time.strftime('%Y-%m-%d %H:%M:%S'))
        save_json(ROOT / 'runs/queue_status.json', status)
        report()
        command = [sys.executable, '-u', str(ROOT / 'run.py'), *args,
                   '--algorithm-profile', 'legacy_source',
                   '--checkpoint-bert', '--resume', '--output', str(ROOT / 'runs' / name)]
        with (ROOT / 'runs' / (name + '.log')).open('a', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=str(ROOT), env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(status='failed', exit_code=result.returncode)
            save_json(ROOT / 'runs/queue_status.json', status)
            report()
            raise SystemExit(result.returncode)
        status['completed'].append(name)
        report()
    status.update(status='complete', finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    save_json(ROOT / 'runs/queue_status.json', status)
    report()


if __name__ == '__main__':
    main()
