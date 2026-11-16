"""A semantic test of causal message direction, independent of source parity."""
import torch
from model.CGE import CGEConv


def main():
    torch.use_deterministic_algorithms(True)
    layer = CGEConv(in_channels={'event': 4}, out_channels=4,
                    metadata=(['event'], [('event', 'intra', 'event')]), heads=1)
    with torch.no_grad():
        layer.proj['event'].weight.copy_(torch.eye(4))
        layer.proj['event'].bias.zero_()
        for parameter in list(layer.lin_src.values()) + list(layer.lin_dst.values()):
            parameter.zero_()
    x = torch.tensor([[2.,3.,4.,5.], [7.,11.,13.,17.]], requires_grad=True)
    edges = {('event', 'intra', 'event'): torch.tensor([[0], [1]])}
    old = layer({'event': x}, edges)['event']
    assert torch.equal(old[0], x[1]) and old[1].abs().sum() == 0
    layer.incoming_orientation = True
    actual = layer({'event': x}, edges)['event']
    assert torch.equal(actual[1], x[0]) and actual[0].abs().sum() == 0
    actual[1].sum().backward()
    assert torch.equal(x.grad, torch.tensor([[1.,1.,1.,1.], [0.,0.,0.,0.]]))
    print('PASS: source mode reverses 0->1; incoming mode sends node 0 into node 1; gradient follows the directed dependency')


if __name__ == '__main__':
    main()
