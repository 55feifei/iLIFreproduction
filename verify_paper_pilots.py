"""Reload the three short paper-algorithm pilots; never launches training."""
import json
import os
import subprocess
import sys
from common import ROOT, save_json
from algorithm_config import PAPER, PAPER_SUFFIX, validate_contract, selection_metric


def main():
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='209')
    checked = []
    names = ['esc_direction_pilot', 'esc_existence_pilot', 'maven_direction_pilot']
    with (ROOT / 'artifacts/paper_algorithm_pilot_reload.log').open('w', encoding='utf-8') as log:
        for prefix in names:
            name = prefix + '_incoming_paper' + PAPER_SUFFIX
            directory = ROOT / 'runs' / name
            config = json.loads((directory / 'configuration.json').read_text(encoding='utf-8'))
            validate_contract(config)
            assert config['algorithm_profile'] == PAPER and config['partial_experiment']
            assert config['epochs'] == 1 and config['limit_train'] <= 8 and config['limit_eval'] <= 4
            fold = directory / 'fold_1'
            result = json.loads((fold / 'result.json').read_text(encoding='utf-8'))
            predictions = ROOT / ('artifacts/reloaded_%s_fold_1.json' % name)
            for script, arguments in [('infer.py', ['--checkpoint', fold / 'best.pt', '--output', predictions]),
                ('verify_reload.py', ['--checkpoint', fold / 'best.pt', '--prediction-file', predictions])]:
                subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)],
                    cwd=str(ROOT), env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            evidence = json.loads((ROOT / ('artifacts/reload_verification_%s_fold_1.json' % name)).read_text(encoding='utf-8'))
            assert evidence['metrics_match'] and evidence['causal_graphs_match']
            checked.append(dict(experiment=name, selection_metric=selection_metric(config),
                best_epoch=result['epoch'], documents=evidence['matched_documents'],
                predictions_metrics_and_graphs_reload_identically=True))
    save_json(ROOT / 'artifacts/paper_algorithm_pilots.json',
        dict(status='passed', algorithm_profile=PAPER, benchmark_training=False, pilots=checked))
    print('PASS: all 3 real-data pilots independently reloaded with identical predictions, ECGs and metrics')


if __name__ == '__main__':
    main()
