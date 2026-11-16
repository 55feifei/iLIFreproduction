"""Independent, small checks of the paper algorithm; no benchmark training."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from algorithm_config import (PAPER, PAPER_V3, PAPER_V2, PAPER_V1, paper_contract, build_model,
                              validate_resume, validate_contract, selection_metric, model_args)
from fusion_config import fusion_settings
from common import scores, save_json, ROOT
from run import evaluate
from analyze_results import verify_ecg


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 768)

    def forward(self, ids, mask):
        return (self.embedding(ids),)

    def resize_token_embeddings(self, size):
        if size != self.embedding.num_embeddings:
            self.embedding = nn.Embedding(size, 768)


class SpyGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, features, graph):
        self.inputs.append((features['event'].detach().clone(), graph))
        return {'event': features['event'] + 1.}


INTRA, INTER = ('event', 'intra', 'event'), ('event', 'inter', 'event')


def configuration(dataset='esc', task='direction', version=PAPER):
    return dict(dataset=dataset, task=task, model_path='test-only', vocab_size=32,
                **dict(fusion_settings('paper'), **paper_contract(dataset, version)))


def make_model(config=None):
    with patch('model.model.BertModel.from_pretrained', side_effect=lambda *a, **k: Encoder()):
        return build_model(config or configuration(), torch.device('cpu'))


def inputs(sentences=12):
    return dict(enc_input_ids=torch.tensor([[1, 2, 3, 4]]),
        enc_mask_ids=torch.ones((1, 4), dtype=torch.long),
        node_event=[[0, [1, 2]], [0, [2, 3]]], t1_pos=[[0]], t2_pos=[[1]],
        target=[[1]], rel_type=[[1]], event_pairs=[[0, 1]], sentence_count=sentences)


def scripted(model, rounds=24, stable=False):
    rows = [[0., 4., 0.] if stable or i % 2 == 0 else [0., 0., 4.] for i in range(rounds)]
    model.script_logits = nn.Parameter(torch.tensor(rows))
    model.script_calls = 0

    def predict(context, nodes, pairs):
        index = model.script_calls
        model.script_calls += 1
        return model.script_logits[index:index + 1]
    model.predict = predict
    model.CGE = SpyGraph()
    model.threshold = 0


class AlgorithmChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(209)

    def test_confidence_boundary_training_and_evaluation(self):
        logits = torch.log(torch.tensor([[.25, .5, .25], [.34, .35, .31]]))
        pairs, types = [[0, 1], [1, 2]], [[1, 0]]
        # v3 keeps ω in both modes. Confidence equal to ω stays; the lower one drops.
        model = make_model(configuration(version=PAPER_V3))
        model.rate = model.w = .5
        for training in (True, False):
            model.train(training)
            graph = model.get_graphedge_index(logits, pairs, types)
            self.assertEqual(list(graph), [('event', 'inter', 'event')])
            self.assertEqual(graph[('event', 'inter', 'event')].tolist(), [[0], [1]])
        # v4 training matches author --rate 0: both causal argmax edges, no ω.
        # Evaluation still drops confidence below ω and keeps the equal boundary.
        current = make_model()
        self.assertEqual(current.rate, 0.)
        self.assertEqual(current.w, .6)
        current.w = .5
        current.train(True)
        trained = current.get_graphedge_index(logits, pairs, types)
        self.assertEqual(trained[('event', 'inter', 'event')].tolist(), [[0], [1]])
        self.assertEqual(trained[('event', 'intra', 'event')].tolist(), [[1], [2]])
        current.eval()
        evaluated = current.get_graphedge_index(logits, pairs, types)
        self.assertEqual(list(evaluated), [('event', 'inter', 'event')])
        self.assertEqual(evaluated[('event', 'inter', 'event')].tolist(), [[0], [1]])

    def test_directed_adjacency_difference(self):
        model = make_model()
        before = {('event', 'intra', 'event'): torch.tensor([[0, 1], [1, 2]])}
        after = {('event', 'intra', 'event'): torch.tensor([[1, 2], [0, 1]])}
        self.assertEqual(model.graph_difference(before, after, 3), 4)
        self.assertEqual(model.graph_difference({}, before, 3), 2)
        self.assertEqual(model.graph_difference(before, before, 3), 0)
        self.assertEqual(model.graph_difference({}, {}, 3), 0)

    def test_document_cap_initialization_and_no_unused_update(self):
        for dataset, sentence_count, expected in [('esc', 26, 9), ('esc', 3, 3), ('maven', 25, 19)]:
            model = make_model(configuration(dataset))
            scripted(model)
            loss, _ = model(**inputs(sentence_count))
            self.assertEqual(model.last_graph_updates, expected)
            self.assertEqual(model.script_calls, expected + 1)
            self.assertEqual(len(model.CGE.inputs), expected)
            h = model.pretrained_model(inputs()['enc_input_ids'], inputs()['enc_mask_ids'])[0]
            self.assertTrue(torch.equal(model.CGE.inputs[0][0], h[0, [1, 2]].detach()))
            messages = model.CGE.inputs[0][1]
            self.assertEqual(set(messages), {INTER, INTRA})
            self.assertEqual(messages[INTER].tolist(), [[0], [1]])
            self.assertEqual(messages[INTRA].tolist(), [[0, 1], [0, 1]])
            expected_loss = sum(model.focal_loss(model.script_logits[i:i + 1], torch.tensor([1])) / i
                                for i in range(1, expected + 1))
            self.assertTrue(torch.allclose(loss, expected_loss))
            loss.backward()
            self.assertEqual(float(model.script_logits.grad[0].abs().sum()), 0.)
            self.assertGreater(float(model.script_logits.grad[1:].abs().sum()), 0.)

    def test_stop_after_first_unchanged_graph(self):
        model = make_model()
        scripted(model, stable=True)
        model(**inputs())
        self.assertEqual(model.last_graph_updates, 1)
        self.assertEqual(model.script_calls, 2)

    def record_differences(self, model):
        differences = []
        def predict(context, nodes, pairs):
            differences.append(nodes[pairs[:, 0]] - nodes[pairs[:, 1]])
            return context.new_tensor([[5., 0., 0.]])
        model.predict = predict
        model.CGE = SpyGraph()
        model(**inputs())
        return differences

    def test_empty_ecg_still_propagates_intra_self_loops(self):
        model = make_model()
        differences = self.record_differences(model)
        self.assertEqual(len(model.CGE.inputs), 1)
        self.assertEqual(list(model.CGE.inputs[0][1]), [INTRA])
        self.assertEqual(model.CGE.inputs[0][1][INTRA].tolist(), [[0, 1], [0, 1]])
        self.assertEqual(len(differences), 2)
        self.assertTrue(torch.allclose(differences[1], differences[0], atol=1e-6))
        self.assertGreater(float(differences[1].abs().sum()), 0.)

    def test_retired_v1_empty_graph_uses_zero_neighborhood(self):
        model = make_model(configuration(version=PAPER_V1))
        differences = self.record_differences(model)
        self.assertEqual(len(model.CGE.inputs), 0)
        self.assertEqual(len(differences), 2)
        self.assertTrue(torch.equal(differences[1], torch.zeros_like(differences[1])))

    def test_self_loops_prevent_chain_collapse(self):
        chain = {INTRA: torch.tensor([[0, 1, 2], [1, 2, 3]])}
        nonzero = {}
        for version in (PAPER, PAPER_V1):
            torch.manual_seed(209)
            model = make_model(configuration(version=version))
            z = torch.randn(5, 768)
            with torch.no_grad():
                for _ in range(5):
                    z = model.CGE({'event': z}, model.message_graph(chain, 5, z.device))['event']
            nonzero[version] = int((z.abs().sum(-1) > 0).sum())
        self.assertEqual(nonzero[PAPER], 5)
        self.assertEqual(nonzero[PAPER_V1], 0)

    def test_focal_formula_and_mean_reduction(self):
        model = make_model()
        logits = torch.tensor([[.2, 1.4, -.1], [.7, -.2, .4]])
        target = torch.tensor([1, 0])
        probability = logits.softmax(-1)[torch.arange(2), target]
        expected = -(torch.tensor([.75, .25]) * (1 - probability)**2 * probability.log()).mean()
        self.assertTrue(torch.allclose(model.focal_loss(logits, target), expected))
        # Mean reduction: duplicating the event pairs leaves the loss unchanged.
        self.assertTrue(torch.allclose(model.focal_loss(logits.repeat(2, 1), target.repeat(2)), expected))
        self.assertIsInstance(model.mlp[1], nn.LeakyReLU)

    def test_real_attention_gradients_and_checkpointing(self):
        model = make_model()
        model.train()
        model.checkpoint_mlp = True
        with torch.no_grad():
            model.mlp[-1].bias[1] = 5.
        loss, logits = model(**inputs())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in (model.pretrained_model, model.mlp, model.CGE):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertIsNone(model.CGE.proj['event'].bias)

    def test_checkpoint_forward_gradients_and_rng_are_equivalent(self):
        reference, recomputed = make_model(), make_model()
        with torch.no_grad():
            reference.mlp[-1].bias[1] = 5.
        recomputed.load_state_dict(reference.state_dict())
        results = []
        for model, enabled in [(reference, False), (recomputed, True)]:
            model.train()
            model.checkpoint_mlp = enabled
            torch.manual_seed(1009)
            loss, logits = model(**inputs())
            loss.backward()
            results.append((loss.detach(), logits.detach(), torch.get_rng_state(),
                {name: parameter.grad for name, parameter in model.named_parameters()}))
        for left, right in zip(results[0][:3], results[1][:3]):
            self.assertTrue(torch.equal(left, right))
        for name, gradient in results[0][3].items():
            other = results[1][3][name]
            if gradient is None:
                self.assertIsNone(other)
            else:
                self.assertTrue(torch.allclose(gradient, other, atol=1e-6, rtol=1e-5), name)

    def test_binary_task_uses_two_classes_and_paper_output(self):
        config = configuration(task='existence')
        model = make_model(config)
        self.assertEqual(model.mlp[-1].out_features, 2)
        self.assertEqual(model.focal_loss.alpha.tolist(), [.25, .75])
        loss, logits = model(**inputs())
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(selection_metric(config), 'existence')

    def test_argmax_output_thresholded_graph_and_standard_direction_metric(self):
        f = SimpleNamespace(doc_id='test', enc_input_ids=torch.ones((1, 3), dtype=torch.long),
            enc_mask_ids=torch.ones((1, 3), dtype=torch.long), node_event=[], t1_pos=[0, 1],
            t2_pos=[1, 2], target=[1, 1], rel_type=[0, 1], event_pairs=[[0, 1], [1, 2]], sentences=['a', 'b'])
        for version, expected_pred in [(PAPER, [1, 2]), (PAPER_V3, [1, 2]), (PAPER_V2, [0, 2])]:
            model = make_model(configuration(version=version))
            def forward(**kwargs):
                return torch.tensor(0.), torch.log(torch.tensor([[.25, .5, .25], [.1, .1, .8]]))
            model.forward = forward
            metrics, documents = evaluate(model, [f], torch.device('cpu'))
            self.assertEqual(documents[0]['argmax_pred'], [1, 2])
            self.assertEqual(documents[0]['pred'], expected_pred)
            self.assertEqual(documents[0]['final_ecg'], {'inter': [[2, 1]]})
            self.assertEqual(verify_ecg(documents[0]), [0, 2])
            self.assertEqual(metrics['direction']['all']['predicted'], sum(p > 0 for p in expected_pred))
        result = scores([1, 1], [1, 2], [0, 1])['direction']['all']
        self.assertEqual((result['tp'], result['predicted'], result['gold']), (1, 2, 2))
        self.assertEqual(result['f1'], .5)

    def test_profiles_resume_protection_and_model_reconstruction(self):
        config = configuration()
        validate_resume(config, copy.deepcopy(config))
        with self.assertRaises(ValueError):
            validate_resume(dict(dataset='esc'), config)
        for key, value in [('class_weight', .5), ('relation_threshold', 0.), ('train_relation_threshold', .6),
                           ('initialization', 'projection'), ('self_loops', False),
                           ('prediction_protocol', 'thresholded_final_ecg')]:
            modified = dict(config, **{key: value})
            with self.assertRaises(ValueError):
                validate_resume(config, modified)
        for version in (PAPER_V3, PAPER_V2, PAPER_V1):
            retired = configuration(version=version)
            validate_contract(retired)
            with self.assertRaises(ValueError):
                validate_resume(retired, config)
            self.assertEqual(selection_metric(retired), 'direction')
        self.assertNotIn('train_relation_threshold', configuration(version=PAPER_V3))
        self.assertEqual(selection_metric(config), 'direction')
        self.assertEqual(selection_metric(dict(dataset='esc', task='direction')), 'source_direction')
        current_args = model_args(config, 'cpu')
        self.assertEqual(current_args.class_weight, .75)
        self.assertEqual(current_args.rate, 0.)
        self.assertEqual(current_args.w, .6)
        retired_args = model_args(configuration(version=PAPER_V3), 'cpu')
        self.assertEqual(retired_args.rate, .6)
        self.assertEqual(retired_args.w, .6)


def main():
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AlgorithmChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = dict(status='passed' if result.wasSuccessful() else 'failed',
        algorithm_profile=PAPER, tests=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), device='cpu', benchmark_training=False)
    save_json(ROOT / 'artifacts/paper_algorithm_checks.json', evidence)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
