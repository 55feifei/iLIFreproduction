"""Check full-model tail ordering and gates without training or CUDA allocation."""
import copy
from pathlib import Path
import tempfile

from queue_full_model_tail import ROOT, build_plan, predecessor_action, exclusive_lock


def main():
    batch = ROOT / 'runs/esc_local_ablations_v4_20261105'
    plan = build_plan(batch)
    assert [job['task'] for job in plan['jobs']] == ['direction', 'existence']
    assert plan['full_fold_models'] == 10 and len(plan['required_completed']) == 6
    for job in plan['jobs']:
        command = job['command']
        assert command[command.index('--ablation') + 1] == 'none'
        assert command[command.index('--epochs') + 1] == '30'
        first = command.index('--folds') + 1
        assert command[first:first + 5] == ['1', '2', '3', '4', '5']
        assert command[command.index('--seed') + 1] == '209'
        assert Path(command[command.index('--output') + 1]).parent == batch
        assert '--resume' in command and '--no-refresh-report' in command
        assert '--limit-train' not in command and '--limit-eval' not in command
    required = plan['required_completed']
    state = dict(status='running', completed=required)
    assert predecessor_action(state, required) == 'waiting_for_ablations'
    state['status'] = 'paused'
    assert predecessor_action(state, required) == 'paused'
    state['status'] = 'failed'
    assert predecessor_action(state, required) == 'blocked'
    state['status'] = 'complete'
    assert predecessor_action(state, required) == 'ready'
    incomplete = copy.deepcopy(state)
    incomplete['completed'].pop()
    try:
        predecessor_action(incomplete, required)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Incomplete ablations were accepted')
    with tempfile.TemporaryDirectory(prefix='ilif_tail_check_') as directory:
        path = Path(directory) / 'tail.lock'
        with exclusive_lock(path):
            try:
                with exclusive_lock(path):
                    raise AssertionError('Duplicate tail dispatcher was accepted')
            except RuntimeError:
                pass
        with exclusive_lock(path):
            pass
    print('PASS: two full-model jobs; five folds x30 epochs; ablation completion gate; pause/failure gate; exclusive dispatcher')


if __name__ == '__main__':
    main()
