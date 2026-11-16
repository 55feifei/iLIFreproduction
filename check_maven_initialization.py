"""Compare vocabulary expansion and RNG state to the author's MAVEN model."""
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import BertConfig, BertModel
from common import AUTHOR
from adapted_model import ReproductionModel


def tiny_bert(*args, **kwargs):
    return BertModel(BertConfig(vocab_size=100, hidden_size=768,
        num_hidden_layers=1, num_attention_heads=4, intermediate_size=96))


class Tokenizer:
    def __len__(self):
        return 102


def main():
    package_path = AUTHOR / 'model'
    spec = importlib.util.spec_from_file_location('author_network', package_path / '__init__.py',
                                                submodule_search_locations=[str(package_path)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    args = SimpleNamespace(device=torch.device('cpu'), model_name='synthetic-init-check',
        n_last=768, mlp_size=48, mlp_drop=.4, no_of_classes=3, rate=0., w=.6,
        max_iteration=19, threshold=2, min_iteration=2, in_channels={'event': 768},
        metadata=(['event'], [('event','intra','event'), ('event','inter','event')]),
        num_heads=4, beta_intra=.3, beta_inter=.7, gamma=2., loss_type='focal', resize_vocab_size=102)
    torch.manual_seed(209)
    with patch('author_network.model.BertModel.from_pretrained', side_effect=tiny_bert):
        original = package.bertCSRModel(args, Tokenizer())
    expected_rng = torch.get_rng_state()
    torch.manual_seed(209)
    with patch('model.model.BertModel.from_pretrained', side_effect=tiny_bert):
        actual = ReproductionModel(args)
    assert torch.equal(expected_rng, torch.get_rng_state()), 'Vocabulary expansion changed RNG consumption'
    original_state, actual_state = original.state_dict(), actual.state_dict()
    assert original_state.keys() == actual_state.keys()
    for name, expected in original_state.items():
        assert torch.equal(expected, actual_state[name]), name
    print('PASS: MAVEN vocabulary expansion, every initialized parameter, and RNG consumption match author model exactly')


if __name__ == '__main__':
    main()
