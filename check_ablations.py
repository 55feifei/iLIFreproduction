"""Current paper model checks for Table 3; no dataset, BERT download or CUDA needed."""
import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
import torch
from unittest.mock import patch
from algorithm_config import PAPER, validate_resume, validate_contract
from common import ROOT, save_json
from check_paper_algorithm import configuration, make_model, inputs, scripted, INTRA, INTER
from ablation_config import VARIANTS, TASKS
from analyze_results import verify_ecg
from run import evaluate
from types import SimpleNamespace
from ablation_preflight import sha256
import analyze_ablations as reporting


def config(variant, task='direction'):
    return dict(configuration(task=task), ablation=variant)


class AblationChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(209)

    def test_direction_is_removed_only_from_graph(self):
        pairs, types = [[0, 1], [1, 2], [0, 2]], [[0, 1, 0]]
        logits = torch.tensor([[0., 5., 0.], [0., 0., 5.], [5., 0., 0.]])
        for training in (True, False):
            model = make_model(config('no_direction')).train(training)
            graph = model.get_graphedge_index(logits, pairs, types)
            self.assertEqual(set(map(tuple, graph[INTRA].t().tolist())), {(0, 1), (1, 0)})
            self.assertEqual(set(map(tuple, graph[INTER].t().tolist())), {(1, 2), (2, 1)})
            reverse = model.get_graphedge_index(logits[:, [0, 2, 1]], pairs, types)
            self.assertEqual(model.graph_difference(graph, reverse, 3), 0)
            self.assertEqual(model.mlp[-1].out_features, 3)

    def test_undirected_adjacency_change_counts_both_entries(self):
        model = make_model(config('no_direction'))
        graph = {INTRA: torch.tensor([[0, 1], [1, 0]])}
        self.assertEqual(model.graph_difference({}, graph, 3), 2)
        self.assertEqual(model.graph_difference(graph, {}, 3), 2)
        model.message_graph(graph, 3, 'cpu')
        self.assertEqual(graph[INTRA].shape[1], 2, 'Message-only self-loops must not change the causal graph')

    def test_one_edge_type_and_no_two_branch_fusion(self):
        model = make_model(config('no_heterogeneity')).eval()
        logits = torch.tensor([[0., 5., 0.], [0., 0., 5.]])
        graph = model.get_graphedge_index(logits, [[0, 1], [1, 2]], [[0, 1]])
        self.assertEqual(list(graph), [INTRA])
        self.assertEqual(graph[INTRA].t().tolist(), [[0, 1], [2, 1]])
        messages = model.message_graph(graph, 3, 'cpu')
        z = torch.randn(3, 768)
        first = model.CGE({'event': z}, messages)['event']
        model.CGE.beta_intra, model.CGE.beta_inter = .01, .99
        second = model.CGE({'event': z}, messages)['event']
        torch.testing.assert_close(first, second)

    def test_no_iteration_exactly_one_update_and_only_final_loss(self):
        model = make_model(config('no_iteration'))
        model.record_trace = True
        scripted(model)  # Opposite directions never converge with threshold=0.
        loss, logits = model(**inputs())
        self.assertEqual(len(model.CGE.inputs), 1)
        self.assertEqual(model.script_calls, 2)
        self.assertEqual(len(model.iteration_trace), 2)
        torch.testing.assert_close(loss, model.focal_loss(logits, torch.tensor([1])))
        loss.backward()
        self.assertEqual(float(model.script_logits.grad[0].abs().sum()), 0.)
        self.assertGreater(float(model.script_logits.grad[1].abs().sum()), 0.)

    def test_iteration_ablation_does_not_limit_other_variants(self):
        for variant in ('none', 'no_direction', 'no_heterogeneity'):
            model = make_model(config(variant))
            # Alternate absent and present edges; undirected direction flips alone
            # would correctly have zero topology difference.
            scripted(model)
            with torch.no_grad():
                model.script_logits[1::2] = torch.tensor([4., 0., 0.])
            model(**inputs(sentences=4))
            self.assertEqual(len(model.CGE.inputs), 4, variant)

    def test_actual_graph_gradients_for_both_tasks_and_all_variants(self):
        for variant in VARIANTS:
            for task in TASKS:
                with self.subTest(variant=variant, task=task):
                    model = make_model(config(variant, task)).train()
                    model.record_trace = True
                    with torch.no_grad():
                        model.mlp[-1].bias[1] = 4.
                    sample = inputs(sentences=4)
                    loss, logits = model(**sample)
                    self.assertTrue(torch.isfinite(loss))
                    self.assertEqual(logits.shape, (1, 2 if task == 'existence' else 3))
                    expected = sum(model.focal_loss(pred, torch.tensor([1])) / index
                        for index, pred in enumerate(model.iteration_trace[1:], 1))
                    torch.testing.assert_close(loss.detach(), expected)
                    loss.backward()
                    for module in (model.pretrained_model, model.mlp, model.CGE):
                        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            and p.grad.abs().sum() > 0 for p in module.parameters()))
                    if variant == 'no_iteration':
                        self.assertEqual(model.last_graph_updates, 1)
                    if variant == 'no_heterogeneity':
                        self.assertIsNone(model.CGE.lin_src['event__inter__event'].grad)

    def test_threshold_protocol_is_unchanged(self):
        logits = torch.log(torch.tensor([[.25, .5, .25]]))
        for variant in VARIANTS:
            model = make_model(config(variant))
            self.assertTrue(model.train().get_graphedge_index(logits, [[0, 1]], [[1]]))
            self.assertEqual(model.eval().get_graphedge_index(logits, [[0, 1]], [[1]]), {})
            self.assertEqual(model.prediction_protocol, 'final_argmax')
            self.assertEqual(model.self_loops, 'intra')

    def test_exported_graph_matches_ablation_and_argmax_output(self):
        for variant in VARIANTS:
            model = make_model(config(variant))
            with torch.no_grad():
                model.mlp[-1].bias[1] = 4.
            sample = inputs(sentences=2)
            feature = SimpleNamespace(doc_id='test-only', sentences=['s1', 's2'],
                **{key: sample[key] for key in ['enc_input_ids', 'enc_mask_ids', 'node_event', 'event_pairs']},
                t1_pos=[0], t2_pos=[1], target=[1], rel_type=[1])
            _, records = evaluate(model, [feature], torch.device('cpu'))
            self.assertEqual(records[0]['pred'], records[0]['argmax_pred'])
            verify_ecg(records[0], variant)

    def test_saved_ablation_rebuilds_and_cannot_resume_other_variant(self):
        for variant in VARIANTS:
            original = config(variant)
            model = make_model(original).eval()
            restored = make_model(original).eval()
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(model(**inputs())[1], restored(**inputs())[1])
            validate_resume(original, original)
            for other in VARIANTS:
                if other != variant:
                    with self.assertRaises(ValueError):
                        validate_resume(original, config(other))
        with self.assertRaises(ValueError):
            validate_contract(config('typo'))


class AblationReportChecks(unittest.TestCase):
    def test_stale_checkpoint_or_configuration_rejects_reload_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            fold = directory / 'fold_1'
            fold.mkdir()
            files = {'best.pt': 'weights', 'best_predictions.json': '[{}]',
                     'result.json': '{}', 'split.json': '{}', 'reloaded_predictions.json': '{}'}
            for name, content in files.items():
                (fold / name).write_text(content, encoding='utf-8')
            (directory / 'configuration.json').write_text('{}', encoding='utf-8')
            report = dict(metrics_match=True, causal_graphs_match=True, matched_documents=1,
                configuration_sha256=sha256(directory / 'configuration.json'),
                sha256={name: sha256(fold / name) for name in files if name != 'reloaded_predictions.json'},
                prediction_sha256=sha256(fold / 'reloaded_predictions.json'))
            save_json(fold / 'reload_verification.json', report)
            self.assertEqual(reporting.verify_reloads(directory, [dict(fold=1)]), [1])
            (fold / 'best.pt').write_text('different weights', encoding='utf-8')
            with self.assertRaises(AssertionError):
                reporting.verify_reloads(directory, [dict(fold=1)])
            (fold / 'best.pt').write_text('weights', encoding='utf-8')
            (directory / 'configuration.json').write_text('{"ablation":"changed"}', encoding='utf-8')
            with self.assertRaises(AssertionError):
                reporting.verify_reloads(directory, [dict(fold=1)])

    def test_five_trained_folds_without_reloads_are_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / 'runs' / reporting.experiment_name('direction', 'none')
            directory.mkdir(parents=True)
            save_json(directory / 'configuration.json', dict(algorithm_profile=PAPER, dataset='esc',
                                                            task='direction', ablation='none'))
            artifact = dict(configuration={}, verified_folds=[dict(fold=i) for i in range(1, 6)], complete=True,
                            mean_fold={'direction': {'all': {'p': .4, 'r': .5, 'f1': .44}}})
            with patch.object(reporting, 'ROOT', root), \
                 patch.object(reporting, 'inspect_experiment', return_value=artifact), \
                 patch.object(reporting, 'plot_learning'), \
                 patch.object(reporting, 'verify_reloads', return_value=[]), \
                 contextlib.redirect_stdout(io.StringIO()):
                reporting.main(['--runs-root', str(root / 'runs')])
            self.assertEqual(len((root / 'artifacts/ablation_comparison.csv').read_text(encoding='utf-8-sig').splitlines()), 1)
            self.assertFalse(artifact['complete'])
            self.assertIn('未完成核验', (root / 'ABLATION_RESULTS.md').read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(2)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case)
                               for case in (AblationChecks, AblationReportChecks))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = dict(status='passed' if result.wasSuccessful() else 'failed',
        algorithm_profile=PAPER, variants=list(VARIANTS), tasks=list(TASKS),
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        torch_version=torch.__version__, device='cpu', benchmark_training=False,
        encoder='tiny test embedding; real project graph encoder, classifier and focal loss')
    save_json(ROOT / 'artifacts/ablation_checks.json', evidence)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
