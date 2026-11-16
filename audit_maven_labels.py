"""Compare every prepared MAVEN label/direction with raw event-level relations."""
import json
from collections import Counter
from common import ROOT, AUTHOR, load_features, save_json


def main():
    raw = {}
    raw_ids = {}
    for filename in ['train', 'valid']:
        documents = [json.loads(line) for line in (AUTHOR / ('data/maven_data/%s.jsonl' % filename)).open(encoding='utf-8') if line.strip()]
        raw_ids[filename] = {document['id'] for document in documents}
        assert len(raw_ids[filename]) == len(documents)
        for document in documents:
            assert document['id'] not in raw
            raw[document['id']] = document
    report, seen = {}, {}
    for split in ['train', 'dev', 'test']:
        features = load_features(ROOT / ('data/maven/%s.pkl' % split))
        counts = Counter()
        pair_count = 0
        seen[split] = set()
        for feature in features:
            doc = raw[feature.doc_id]
            assert feature.doc_id not in seen[split]
            seen[split].add(feature.doc_id)
            events = sorted(doc['events'], key=lambda event: (
                event['mention'][0]['sent_id'], event['mention'][0]['offset'][0]))
            event_ids = [event['id'] for event in events]
            assert len(event_ids) == len(set(event_ids)) == len(feature.node_event)
            expected = {}
            for kind, pairs in doc['causal_relations'].items():
                assert kind in ['CAUSE', 'PRECONDITION']
                for pair in pairs:
                    key = tuple(pair)
                    assert key not in expected, (feature.doc_id, 'duplicate relation', key)
                    expected[key] = kind
            observed, unordered = set(), set()
            for pair, label, relation_type in zip(feature.event_pairs, feature.target, feature.rel_type):
                a, b = map(int, pair)
                assert a != b and 0 <= a < len(events) and 0 <= b < len(events)
                key = tuple(sorted((a, b)))
                assert key not in unordered
                unordered.add(key)
                forward, reverse = (event_ids[a], event_ids[b]), (event_ids[b], event_ids[a])
                assert int(relation_type) == int(events[a]['mention'][0]['sent_id'] != events[b]['mention'][0]['sent_id'])
                label = int(label)
                assert label in [0, 1, 2]
                if label:
                    directed = forward if label == 1 else reverse
                    assert directed in expected, (feature.doc_id, directed, label)
                    assert directed not in observed
                    observed.add(directed)
                    counts[expected[directed]] += 1
                else:
                    assert forward not in expected and reverse not in expected, (feature.doc_id, 'positive marked NONE')
            assert len(feature.event_pairs) == len(feature.target) == len(feature.rel_type)
            assert len(unordered) == len(events) * (len(events) - 1) // 2
            assert observed == set(expected), (feature.doc_id, 'missing/extra causal relations')
            pair_count += len(feature.target)
        report[split] = dict(documents=len(features), event_pairs=pair_count,
                             positive_relations_by_raw_type=dict(counts), positive_relations=sum(counts.values()))
    assert seen['train'].isdisjoint(seen['dev'])
    assert seen['train'] | seen['dev'] == raw_ids['train']
    assert seen['test'].issubset(raw_ids['valid'])
    omitted = raw_ids['valid'] - seen['test']
    assert all(len(raw[doc_id]['events']) < 2 and not any(raw[doc_id]['causal_relations'].values()) for doc_id in omitted)
    save_json(ROOT / 'artifacts/maven_raw_label_audit.json', dict(
        passed=True, scope='all prepared event pairs, both raw causal types, exact directions and sentence relation types',
        splits=report, omitted_original_validation_documents=sorted(omitted)))
    print(report)


if __name__ == '__main__':
    main()
