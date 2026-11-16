"""CPU diagnostic for Eq. (4); does not change training or model weights."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from adapted_model import ReproductionModel
from model.CGE import CGEConv, group


def main():
    torch.set_num_threads(1)
    intra_type = ('event', 'intra', 'event')
    inter_type = ('event', 'inter', 'event')
    intra = torch.tensor([[2., 4.]])
    inter = torch.tensor([[10., 20.]])
    actual = group([intra, inter], .3, .7)
    paper = .7 * intra + .3 * inter
    assert torch.allclose(actual, .3 * intra + .7 * inter)
    assert not torch.allclose(actual, paper)

    conv = CGEConv({'event': 2}, 2, (['event'], [intra_type, inter_type]),
                   heads=1, beta_intra=.3, beta_inter=.7)
    conv.incoming_orientation = True
    conv.eval()
    with torch.no_grad():
        conv.proj['event'].weight.copy_(torch.eye(2))
        conv.proj['event'].bias.zero_()
        for parameter in list(conv.lin_src.values()) + list(conv.lin_dst.values()):
            parameter.zero_()
        x = {'event': torch.tensor([[1., 2.], [3., 4.], [5., 6.]])}
        loops = torch.tensor([[0, 1, 2], [0, 1, 2]])
        cross = torch.tensor([[0, 1, 2], [1, 2, 0]])
        normal = conv(x, {intra_type: loops, inter_type: cross})['event']
        reordered = conv(x, {inter_type: cross, intra_type: loops})['event']
    assert not torch.allclose(normal, reordered)

    probe = SimpleNamespace(rate=0., w=.6, training=True, ablation='none')
    edges = ReproductionModel.get_graphedge_index(
        probe, torch.tensor([[0., 10., 0.]]), [[0, 1]], [[1]])
    assert list(edges) == [inter_type]
    # Same insertion used by bertCSRModel.forward for an inter-only graph.
    edges[intra_type] = torch.tensor([[0, 1], [0, 1]])
    assert list(edges) == [inter_type, intra_type]

    evidence = {
        'device': 'cpu', 'training_modified': False,
        'paper_beta_intra': .7, 'paper_beta_inter': .3,
        'current_entrypoint_beta_intra': .3, 'current_entrypoint_beta_inter': .7,
        'group_example': {'intra': intra.tolist(), 'inter': inter.tolist(),
                          'current': actual.tolist(), 'paper': paper.tolist()},
        'relation_order_test': {'intra_then_inter': normal.tolist(),
                                'inter_then_intra': reordered.tolist(),
                                'max_absolute_difference': (normal - reordered).abs().max().item(),
                                'invariant': False},
        'inter_only_after_self_loop_relation_order': [kind[1] for kind in edges],
        'conclusion': 'Paper weight mismatch and positional relation fusion both confirmed.'}
    output = Path(__file__).resolve().parent / 'artifacts/fusion_weight_audit.json'
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == '__main__':
    main()
