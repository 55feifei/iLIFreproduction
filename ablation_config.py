"""Table 3 protocol and experiment names (no training dependencies required)."""
from algorithm_config import PAPER, PAPER_SUFFIX

PAPER_URL = 'https://aclanthology.org/2024.acl-long.210.pdf'
VARIANTS = ('none', 'no_direction', 'no_heterogeneity', 'no_iteration')
TASKS = ('direction', 'existence')
TARGETS = {
    'none': {'direction': [47.9, 47.8, 47.8], 'existence': [59.2, 66.1, 62.5]},
    'no_direction': {'direction': [42.3, 49.3, 45.5], 'existence': [55.9, 63.8, 59.6]},
    'no_heterogeneity': {'direction': [45.5, 47.2, 46.3], 'existence': [56.4, 66.3, 61.0]},
    'no_iteration': {'direction': [43.2, 45.2, 44.2], 'existence': [52.8, 65.9, 58.6]},
}


def experiment_name(task, variant, phase='full'):
    if task not in TASKS or variant not in VARIANTS or phase not in ('pilot', 'full'):
        raise ValueError('Invalid Table 3 experiment')
    base = 'esc_%s_%s_incoming_paper%s' % (task, phase, PAPER_SUFFIX)
    return base if variant == 'none' else base + '_' + variant


def training_command(python, root, output_root, model_path, task, variant, phase,
                     folds=(1, 2, 3, 4, 5), seed=209, threads=4):
    command = [str(python), '-u', str(root / 'run.py'), '--dataset', 'esc',
        '--task', task, '--ablation', variant, '--algorithm-profile', PAPER,
        '--graph-orientation', 'incoming', '--fusion-policy', 'paper',
        '--checkpoint-bert', '--resume', '--model-path', str(model_path),
        '--seed', str(seed), '--threads', str(threads),
        '--output', str(output_root / experiment_name(task, variant, phase))]
    if phase == 'pilot':
        command += ['--epochs', '1', '--folds', '1', '--limit-train', '8', '--limit-eval', '4']
    else:
        command += ['--epochs', '30', '--folds'] + list(map(str, folds))
    command.append('--no-refresh-report')
    return command
