"""Dependency-free checks of planning, pilot gates and queue failure handling."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import run_ablations as queue
from ablation_config import experiment_name


class QueueChecks(unittest.TestCase):
    def test_default_plan_has_matched_baselines_and_forty_fold_models(self):
        plan = queue.build_plan(queue.parse_args([]))
        self.assertEqual(plan['full_fold_models'], 40)
        self.assertEqual(len(plan['jobs']), 16)
        self.assertEqual([job['phase'] for job in plan['jobs']], ['pilot'] * 8 + ['full'] * 8)
        self.assertEqual(len({job['name'] for job in plan['jobs']}), 16)
        for job in plan['jobs']:
            command = job['command']
            self.assertIn('--resume', command)
            self.assertEqual(command[command.index('--ablation') + 1], job['variant'])
            self.assertEqual(command[command.index('--epochs') + 1], '1' if job['phase'] == 'pilot' else '30')
        self.assertEqual(experiment_name('direction', 'none'), 'esc_direction_full_incoming_paper_v4')

    def test_dry_run_requires_no_training_imports_or_files(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as output:
            queue.main(['--dry-run', '--output-root', str(Path(temp) / 'missing')])
            self.assertEqual(json.loads(output.getvalue())['full_fold_models'], 40)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_subset_and_pilots_stay_partial(self):
        plan = queue.build_plan(queue.parse_args(['--pilots-only', '--variants', 'no_iteration', '--tasks', 'existence']))
        self.assertEqual(len(plan['jobs']), 1)
        self.assertEqual(plan['full_fold_models'], 0)
        self.assertEqual(plan['jobs'][0]['phase'], 'pilot')

    def test_invalid_duplicates_fail_before_work(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for args in [['--folds', '1', '1'], ['--variants', 'none', 'none'], ['--threads', '0']]:
                with self.assertRaises(SystemExit):
                    queue.parse_args(args)

    def exercise_queue(self, failure=False, pilots_only=False):
        calls = []
        preflight = types.ModuleType('ablation_preflight')
        preflight.check = lambda *args: dict(status='passed', errors=[])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output_root = root / 'new-runs'
            def run(command, log_path, environment):
                self.assertTrue(output_root.is_dir())
                self.assertEqual(environment['HF_HUB_OFFLINE'], '1')
                calls.append(command)
                if failure and Path(command[2]).name == 'run.py':
                    raise RuntimeError('intentional pilot failure')
            args = ['--output-root', str(output_root), '--model-path', str(root / 'bert'),
                    '--esc-data', str(root / 'data'), '--variants', 'no_iteration', '--tasks', 'direction']
            if pilots_only:
                args += ['--pilots-only']
            with patch.object(queue, 'ROOT', root), patch.object(queue, 'run_logged', side_effect=run), \
                 patch.dict(sys.modules, {'ablation_preflight': preflight}), \
                 patch.dict(queue.os.environ, {}, clear=False), contextlib.redirect_stdout(io.StringIO()):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'pilot failure'):
                        queue.main(args)
                else:
                    queue.main(args)
            status = json.loads((output_root / 'ablation_queue_status_paper_v4.json').read_text(encoding='utf-8'))
            return calls, status

    def test_failed_pilot_blocks_full_training_and_records_failure(self):
        calls, status = self.exercise_queue(failure=True)
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['completed'], [])
        self.assertEqual([Path(command[2]).name for command in calls],
                         ['check_ablation_queue.py', 'check_ablations.py', 'run.py'])

    def test_pilots_only_still_requires_independent_reload(self):
        calls, status = self.exercise_queue(pilots_only=True)
        self.assertEqual([Path(command[2]).name for command in calls],
                         ['check_ablation_queue.py', 'check_ablations.py', 'run.py', 'infer.py', 'verify_reload.py'])
        self.assertEqual(status['status'], 'complete')
        self.assertEqual(len(status['completed']), 1)

    def test_successful_full_queue_reloads_all_five_models_before_report(self):
        calls, status = self.exercise_queue()
        names = [Path(command[2]).name for command in calls]
        self.assertEqual(names.count('run.py'), 2)
        self.assertEqual(names.count('infer.py'), 6)
        self.assertEqual(names.count('verify_reload.py'), 6)
        self.assertEqual(names[-1], 'analyze_ablations.py')
        self.assertEqual(status['status'], 'complete')
        self.assertEqual(len(status['completed']), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
