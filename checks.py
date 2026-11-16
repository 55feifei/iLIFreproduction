"""Checks of scientifically consequential evaluation and graph behavior."""
import importlib.util
import logging
import argparse
from pathlib import Path
import torch
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from types import SimpleNamespace
from unittest.mock import patch
from transformers import BertConfig, BertModel
from torch_sparse import SparseTensor
from common import ROOT, scores
from legacy_utils import transfor3to2, compute_f1
from adapted_model import ReproductionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication', action='store_true')
    cli = parser.parse_args()
    result = scores([1, 2, 0], [2, 2, 1], [0, 1, 1])
    assert result['direction']['all']['predicted'] == 3
    assert result['direction']['all']['tp'] == 1
    assert result['source_direction']['all']['predicted'] == 2
    assert result['existence']['all']['tp'] == 2
    g, p = transfor3to2([1, 2, 0], [2, 2, 1])
    old = compute_f1(g, p, logging.getLogger('test'))
    assert abs(old[2] - result['source_direction']['all']['f1']) < 1e-12
    # Sparse adjacency row is receiver, col is sender for PyG source_to_target.
    adjacency = SparseTensor(row=torch.tensor([0]), col=torch.tensor([1]),
                             sparse_sizes=(2, 2))
    assert adjacency.matmul(torch.tensor([[2.], [7.]])).tolist() == [[7.], [0.]]
    spec = importlib.util.spec_from_file_location('original_model', ROOT.parent / 'reference_source/model/model.py')
    # Import source as its own package for relative imports.
    import sys
    sys.path.insert(0, str(ROOT.parent))
    if cli.publication:
        package_path = ROOT / 'official_publication_source/model'
        package_spec = importlib.util.spec_from_file_location('publication_network', package_path / '__init__.py',
            submodule_search_locations=[str(package_path)])
        package = importlib.util.module_from_spec(package_spec)
        sys.modules[package_spec.name] = package
        package_spec.loader.exec_module(package)
        Original = package.bertCSRModel
        original_patch_path = 'publication_network.model.BertModel.from_pretrained'
    else:
        from reference_source.model.model import bertCSRModel as Original
        original_patch_path = 'reference_source.model.model.BertModel.from_pretrained'
    obj = object.__new__(ReproductionModel)
    torch.nn.Module.__init__(obj)
    obj.device = torch.device('cpu')
    obj.rate, obj.w = 0.0, 0.6
    pairs = [[0, 1], [1, 2], [0, 2], [2, 3]]
    types = [[0, 1, 0, 1]]
    logits = torch.tensor([[0., 5., 0.], [0., 0., 5.], [5., 0., 0.], [0., 0., 0.]])
    for training in [True, False]:
        obj.train(training)
        expected = Original.get_graphedge_index(obj, logits, pairs, types)
        actual = obj.get_graphedge_index(logits, pairs, types)
        assert list(expected) == list(actual)
        assert all(torch.equal(expected[k], actual[k]) for k in expected)
        assert Original.Contrast_pre(obj, logits, logits.flip(0)) == obj.Contrast_pre(logits, logits.flip(0))
    device = torch.device('cuda')
    args = SimpleNamespace(device=device, model_name='synthetic-test-only', n_last=768,
        mlp_size=48, mlp_drop=.4, no_of_classes=3, rate=0., w=.6,
        max_iteration=9, threshold=2, min_iteration=2, in_channels={'event': 768},
        metadata=(['event'], [('event','intra','event'), ('event','inter','event')]),
        num_heads=4, beta_intra=.3, beta_inter=.7, gamma=2., loss_type='focal')
    config = BertConfig(vocab_size=100, hidden_size=768, num_hidden_layers=1,
                        num_attention_heads=4, intermediate_size=96)
    torch.manual_seed(209)
    with patch(original_patch_path, side_effect=lambda *a, **k: BertModel(config)):
        original = Original(args).to(device)
    with patch('model.model.BertModel.from_pretrained', side_effect=lambda *a, **k: BertModel(config)):
        adapted = ReproductionModel(args).to(device)
    with torch.no_grad():
        original.mlp[-1].bias[1] = 2.
    adapted.load_state_dict(original.state_dict())
    adapted.CGE.incoming_orientation = cli.publication
    adapted.pretrained_model.gradient_checkpointing_enable()
    adapted.checkpoint_mlp = True
    inputs = dict(enc_input_ids=torch.randint(0, 100, (2, 12), device=device),
        enc_mask_ids=torch.ones((2, 12), dtype=torch.long, device=device),
        node_event=[[0, [1,2]], [0,[3,5]], [1,[1,3]], [1,[4,5]]],
        t1_pos=[[0,1,0,2]], t2_pos=[[1,2,2,3]], target=[[1,0,2,1]],
        rel_type=types, event_pairs=pairs)
    for training in [True, False]:
        original.train(training)
        adapted.train(training)
        torch.use_deterministic_algorithms(False)
        torch.manual_seed(209)
        loss_a, logits_a = original(**inputs)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(209)
        loss_b, logits_b = adapted(**inputs)
        assert torch.allclose(logits_a, logits_b, atol=1e-6), (logits_a - logits_b).abs().max()
        assert torch.allclose(loss_a, loss_b, atol=1e-6)
        if training:
            # Legacy gather backward is unsupported in strict mode in torch 1.11.
            torch.use_deterministic_algorithms(False)
            loss_a.backward()
            torch.use_deterministic_algorithms(True)
            loss_b.backward()
            reference_parameters = dict(original.named_parameters())
            gradient_differences = []
            for name, parameter in adapted.named_parameters():
                expected_grad = reference_parameters[name].grad
                if expected_grad is not None:
                    assert parameter.grad is not None
                    # Publication COO scatter uses non-deterministic floating
                    # additions. Compare it with CSR within float32 tolerance.
                    atol, rtol = (5e-5, 5e-4) if cli.publication else (1e-5, 1e-4)
                    relative_l2 = float((parameter.grad-expected_grad).norm()/(expected_grad.norm()+1e-12))
                    if not torch.allclose(parameter.grad, expected_grad, atol=atol, rtol=rtol) or (
                            cli.publication and relative_l2 > 5e-4):
                        gradient_differences.append(dict(name=name,
                            max_abs=float((parameter.grad-expected_grad).abs().max()),
                            relative_l2=float((parameter.grad-expected_grad).norm()/(expected_grad.norm()+1e-12)),
                            reference_max=float(expected_grad.abs().max())))
            if gradient_differences:
                print('GRADIENT_DIFFERENCES', gradient_differences, flush=True)
            assert not gradient_differences, 'Gradient comparison failed; inspect measured differences'
            for module in [adapted.pretrained_model, adapted.mlp, adapted.CGE]:
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    print('PASS:', 'publication version' if cli.publication else 'provided later version',
          'metrics, graph/stopping, full forward and gradients; strict CUDA backward with BERT/MLP checkpointing')


if __name__ == '__main__':
    main()
