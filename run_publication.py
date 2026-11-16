"""Primary reproduction follows the official pre-publication graph semantics."""
import argparse
import json
import os
import subprocess
import sys
import time
from common import ROOT, save_json
from report import main as report
from fusion_config import fusion_settings
from algorithm_config import PAPER, TRAINABLE_PROFILES, PAPER_SUFFIX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion-policy', choices=['source', 'paper'], default='paper')
    parser.add_argument('--algorithm-profile', choices=TRAINABLE_PROFILES, default=PAPER)
    parser.add_argument('--pilots-only', action='store_true')
    cli = parser.parse_args()
    suffix = '_paper' if cli.fusion_policy == 'paper' else ''
    if cli.algorithm_profile == PAPER:
        if cli.fusion_policy != 'paper':
            parser.error('The paper algorithm requires paper fusion')
        suffix += PAPER_SUFFIX
    status_path = ROOT / ('runs/queue_status%s.json' % suffix)
    audit = json.loads((ROOT/'artifacts/publication_version_audit.json').read_text(encoding='utf-8'))
    assert all(record['matches'] for record in audit['data'].values())
    queue = [
        ('esc_direction_pilot_incoming', ['--dataset','esc','--task','direction','--epochs','1','--folds','1','--limit-train','8','--limit-eval','4']),
        ('esc_existence_pilot_incoming', ['--dataset','esc','--task','existence','--epochs','1','--folds','1','--limit-train','8','--limit-eval','4']),
        ('maven_direction_pilot_incoming', ['--dataset','maven','--epochs','1','--limit-train','8','--limit-eval','4']),
        ('esc_direction_full_incoming', ['--dataset','esc','--task','direction']),
        ('esc_existence_full_incoming', ['--dataset','esc','--task','existence']),
        ('maven_direction_full_incoming', ['--dataset','maven']),
    ]
    queue = [(name + suffix, args) for name, args in queue]
    if cli.pilots_only:
        queue = queue[:3]
    environment = os.environ.copy()
    environment.update(PYTHONHASHSEED='209', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    status = dict(pid=os.getpid(), python=sys.executable, status='running', completed=[],
                  graph_implementation='official_pre_publication', source_commit=audit['commit'])
    status.update(fusion_settings(cli.fusion_policy))
    status.update(algorithm_profile=cli.algorithm_profile, pilots_only=cli.pilots_only)
    for name, args in queue:
        status.update(current=name, started=time.strftime('%Y-%m-%d %H:%M:%S'))
        save_json(status_path, status)
        report()
        command = [sys.executable,'-u',str(ROOT/'run.py'),*args,'--graph-orientation','incoming',
                   '--fusion-policy',cli.fusion_policy,
                   '--algorithm-profile',cli.algorithm_profile,
                   '--checkpoint-bert','--resume','--output',str(ROOT/'runs'/name)]
        with (ROOT/'runs'/(name+'.log')).open('a',encoding='utf-8') as log:
            result = subprocess.run(command,cwd=str(ROOT),env=environment,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(status='failed',exit_code=result.returncode)
            save_json(status_path, status)
            report()
            raise SystemExit(result.returncode)
        status['completed'].append(name)
        report()
    status.update(status='complete',finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    save_json(status_path,status)
    report()


if __name__ == '__main__':
    main()
