"""Dependency-free checks of the ESC existence input protocol."""
import copy
from types import SimpleNamespace
import unittest
from prepare_esc_existence import convert


class PreparationChecks(unittest.TestCase):
    def feature(self):
        return SimpleNamespace(doc_id='example', target=[0, 1, 2],
            event_pairs=[[0, 1], [1, 2], [0, 2]], rel_type=[0, 1, 1],
            t1_pos=[0, 1, 0], t2_pos=[1, 2, 2],
            enc_input_ids=object(), node_event=[[0, [1, 2]]], sentences=['text'])

    def test_reverse_only_effect_and_preserve_row_alignment(self):
        feature = self.feature()
        result = convert(feature)
        self.assertEqual(result.target, [0, 1, 1])
        self.assertEqual(result.event_pairs, [[0, 1], [1, 2], [2, 0]])
        self.assertEqual(result.t1_pos, [0, 1, 2])
        self.assertEqual(result.t2_pos, [1, 2, 0])
        self.assertEqual(result.rel_type, [0, 1, 1])
        self.assertIs(result.enc_input_ids, feature.enc_input_ids)
        self.assertIs(result.node_event, feature.node_event)

    def test_source_labels_and_pairs_are_not_mutated(self):
        feature = self.feature()
        before = copy.deepcopy((feature.target, feature.event_pairs, feature.t1_pos, feature.t2_pos))
        convert(feature)
        self.assertEqual((feature.target, feature.event_pairs, feature.t1_pos, feature.t2_pos), before)

    def test_invalid_label_or_alignment_fails(self):
        feature = self.feature()
        feature.target[2] = 3
        with self.assertRaises(ValueError):
            convert(feature)
        feature = self.feature()
        feature.rel_type.pop()
        with self.assertRaises(ValueError):
            convert(feature)


if __name__ == '__main__':
    unittest.main(verbosity=2)
