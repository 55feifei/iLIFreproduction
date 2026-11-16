"""Verify document isolation, pair alignment, trigger spans and DAG targets."""
from collections import Counter, deque
import networkx as nx
from common import ROOT, ESC_DATA, load_features, describe, save_json


def directed_edges(feature):
    for pair, label in zip(feature.event_pairs, feature.target):
        a, b = map(int, pair)
        if int(label) == 1:
            yield a, b
        elif int(label) == 2:
            yield b, a


def cycle_check(feature):
    adjacency = [set() for _ in feature.node_event]
    indegree = [0] * len(adjacency)
    for a, b in directed_edges(feature):
        if b not in adjacency[a]:
            adjacency[a].add(b)
            indegree[b] += 1
    queue = deque(i for i, count in enumerate(indegree) if not count)
    visited = 0
    while queue:
        a = queue.popleft()
        visited += 1
        for b in adjacency[a]:
            indegree[b] -= 1
            if not indegree[b]:
                queue.append(b)
    return visited == len(adjacency)


def verify_pairs(features):
    # describe verifies t1/t2 agreement with event_pairs and all trigger ranges.
    describe(features)
    for feature in features:
        pairs = [tuple(sorted(pair)) for pair in feature.event_pairs]
        assert len(set(pairs)) == len(pairs), ('duplicate pair', feature.doc_id)
        for a, b in feature.event_pairs:
            assert 0 <= a < len(feature.node_event)
            assert 0 <= b < len(feature.node_event)
            assert a != b


def main():
    esc = {split: load_features(ESC_DATA / ('EventStoryLine_bert_intra_and_inter_%s_features.pkl' % split))
           for split in ['train', 'dev']}
    report = {'esc': {}, 'maven': {}}
    assert set(str(f.doc_id) for f in esc['train']).isdisjoint(str(f.doc_id) for f in esc['dev'])
    assert set(str(f.topic_id) for f in esc['train']).isdisjoint(str(f.topic_id) for f in esc['dev'])
    for split, features in esc.items():
        verify_pairs(features)
        cyclic = [str(f.doc_id) for f in features if not cycle_check(f)]
        report['esc'][split] = dict(documents=len(features), cyclic_gold_documents=cyclic)
        report['esc'][split]['cycle_evidence'] = {}
        for feature in features:
            if str(feature.doc_id) in cyclic:
                graph = nx.DiGraph()
                graph.add_edges_from(directed_edges(feature))
                report['esc'][split]['cycle_evidence'][str(feature.doc_id)] = {
                    'event_index_cycles': list(nx.simple_cycles(graph)),
                    'node_event': [[int(sid), list(map(int, span))] for sid, span in feature.node_event]}
        # Report supplied-data discrepancies without silently changing labels.
    report['esc']['train_dev_document_overlap'] = 0
    report['esc']['train_dev_topic_overlap'] = 0
    maven = {split: load_features(ROOT / ('data/maven/%s.pkl' % split)) for split in ['train', 'dev', 'test']}
    seen = set()
    for split, features in maven.items():
        verify_pairs(features)
        ids = [str(f.doc_id) for f in features]
        assert len(ids) == len(set(ids))
        assert seen.isdisjoint(ids)
        seen.update(ids)
        report['maven'][split] = dict(documents=len(features), pairs=sum(len(f.target) for f in features),
                                     cyclic_gold_documents=sum(not cycle_check(f) for f in features))
    report['maven']['split_document_overlap'] = 0
    save_json(ROOT / 'artifacts/data_integrity.json', report)
    print(report)


if __name__ == '__main__':
    main()
