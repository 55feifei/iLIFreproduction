"""Check Eq. (4), edge masks, relation-order invariance, gradients and resume guards."""
import json
from pathlib import Path

import torch

from fusion_config import fusion_from_config, fusion_settings, validate_fusion_resume
from model.CGE import CGEConv, fuse_by_type


def main():
    torch.set_num_threads(1)
    torch.manual_seed(209)
    intra = torch.tensor([[2., 4.], [0., 0.], [0., 0.], [0., 0.]], requires_grad=True)
    inter = torch.tensor([[10., 20.], [10., 20.], [8., 16.], [0., 0.]], requires_grad=True)
    masks = {'intra': torch.tensor([True, True, False, False]),
             'inter': torch.tensor([True, True, True, False])}
    output = fuse_by_type({'inter': inter, 'intra': intra}, masks, .7, .3)
    expected = torch.tensor([[4.4, 8.8], [3., 6.], [8., 16.], [0., 0.]])
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    torch.testing.assert_close(intra.grad, torch.tensor([[.7, .7], [.7, .7], [0., 0.], [0., 0.]]))
    torch.testing.assert_close(inter.grad, torch.tensor([[.3, .3], [.3, .3], [1., 1.], [0., 0.]]))
    sole = fuse_by_type({'inter': inter.detach()}, {'inter': masks['inter']}, .7, .3)
    torch.testing.assert_close(sole, inter.detach())

    it, et = ('event', 'intra', 'event'), ('event', 'inter', 'event')
    conv = CGEConv({'event': 2}, 2, (['event'], [it, et]), heads=1,
                   beta_intra=.7, beta_inter=.3, fusion_policy='paper').double()
    conv.incoming_orientation = True
    conv.eval()
    with torch.no_grad():
        conv.proj['event'].weight.copy_(torch.eye(2, dtype=torch.float64))
        conv.proj['event'].bias.zero_()
        for p in list(conv.lin_src.values()) + list(conv.lin_dst.values()):
            p.zero_()
    loops = torch.tensor([[0, 1, 2], [0, 1, 2]])
    cross = torch.tensor([[0, 1, 2], [1, 2, 0]])
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64, requires_grad=True)
    actual = conv({'event': x}, {it: loops, et: cross})['event']
    torch.testing.assert_close(actual, .7 * x + .3 * x[[2, 0, 1]], rtol=1e-12, atol=1e-12)
    actual.sum().backward()
    grads = {name: p.grad.clone() for name, p in conv.named_parameters()}
    input_grad = x.grad.clone()
    conv.zero_grad()
    x.grad = None
    reordered = conv({'event': x}, {et: cross, it: loops})['event']
    reordered.sum().backward()
    torch.testing.assert_close(actual, reordered, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(input_grad, x.grad, rtol=1e-12, atol=1e-12)
    for name, p in conv.named_parameters():
        torch.testing.assert_close(grads[name], p.grad, rtol=1e-12, atol=1e-12)

    legacy = fusion_from_config({})
    paper = fusion_settings('paper')
    assert legacy == fusion_settings('source')
    assert paper['beta_intra'] == .7 and paper['beta_inter'] == .3
    validate_fusion_resume({}, legacy)
    validate_fusion_resume(paper, paper)
    rejected = 0
    for changed in [paper, fusion_settings('source', .7, .3)]:
        try:
            validate_fusion_resume({}, changed)
        except ValueError:
            rejected += 1
    assert rejected == 2
    try:
        fusion_settings('paper', .7, .7)
        raise AssertionError('Invalid weight sum was accepted')
    except ValueError:
        pass
    evidence = dict(status='passed', device='cpu', paper_settings=paper,
                    formula_and_zero_activation_case=True, sole_branch_unscaled=True,
                    missing_both_zero=True, relation_order_forward_and_gradients=True,
                    all_parameter_gradients_match=True, legacy_config_compatible=True,
                    mixed_resume_rejected=True, invalid_weights_rejected=True,
                    self_loop_rule='Self-loops count as their explicitly declared relation type.')
    path = Path(__file__).resolve().parent / 'artifacts/paper_fusion_checks.json'
    path.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
