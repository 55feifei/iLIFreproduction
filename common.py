"""Data compatibility and explicit evaluation for iLIF reproduction."""
import json
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import torch

from project_paths import ROOT, PROJECT, ESC_DATA, AUTHOR


class Features:
    pass


class FeatureUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in ('ESC_features', 'MAVEN_features'):
            return Features
        return super().find_class(module, name)


def load_features(path):
    with open(path, 'rb') as stream:
        return FeatureUnpickler(stream).load()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def batch(feature, device):
    return dict(enc_input_ids=feature.enc_input_ids.to(device),
                enc_mask_ids=feature.enc_mask_ids.to(device),
                node_event=feature.node_event, t1_pos=[feature.t1_pos],
                t2_pos=[feature.t2_pos], target=[feature.target],
                rel_type=[feature.rel_type], event_pairs=feature.event_pairs,
                sentence_count=len(feature.sentences))


def scores(gold, pred, rel_type):
    gold, pred, rel_type = [np.asarray(x) for x in (gold, pred, rel_type)]
    result = {}
    for mode in ('direction', 'existence', 'source_direction'):
        result[mode] = {}
        for scope, mask in [('all', np.ones(len(gold), dtype=bool)),
                            ('intra', rel_type == 0), ('inter', rel_type == 1)]:
            g, p = gold[mask], pred[mask]
            tp = int(((g > 0) & ((p > 0) if mode == 'existence' else (g == p))).sum())
            n_gold = int((g > 0).sum())
            n_pred = int((p > 0).sum())
            if mode == 'source_direction':
                # Released transfor3to2 drops opposite-direction predictions.
                n_pred -= int(((g > 0) & (p > 0) & (g != p)).sum())
            precision = tp / n_pred if n_pred else 0.0
            recall = tp / n_gold if n_gold else 0.0
            result[mode][scope] = dict(p=precision, r=recall,
                f1=2 * tp / (n_pred + n_gold) if n_pred + n_gold else 0.0,
                tp=tp, predicted=n_pred, gold=n_gold)
    return result


def describe(features):
    labels, relations = Counter(), Counter()
    for f in features:
        labels.update(map(int, f.target))
        relations.update(map(int, f.rel_type))
        assert len(f.target) == len(f.event_pairs) == len(f.rel_type)
        assert list(f.t1_pos) == [p[0] for p in f.event_pairs]
        assert list(f.t2_pos) == [p[1] for p in f.event_pairs]
        for event in f.node_event:
            occurrences = [event] if isinstance(event[0], (int, np.integer)) else event
            assert occurrences
            for sentence, (start, end) in occurrences:
                assert 0 <= sentence < len(f.enc_input_ids)
                assert 0 <= start < end <= f.enc_input_ids.shape[1]
    return dict(documents=len(features), labels=dict(labels), relation_types=dict(relations),
                topics=sorted(set(str(getattr(f, 'topic_id', '')) for f in features)),
                events=sum(len(f.node_event) for f in features),
                max_input_shape=list(max(features, key=lambda f: f.enc_input_ids.numel()).enc_input_ids.shape) if features else [],
                max_total_input_tokens=max([f.enc_input_ids.numel() for f in features], default=0),
                max_pairs=max([len(f.target) for f in features], default=0))
