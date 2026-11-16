"""Audit a completed MAVEN checkpoint and predictions without interrupting training.

This verifies saved metadata and prediction/data consistency. Independent model
reload remains a separate final-stage check; an interim score is not a main-table
result before all 30 epochs finish.
"""
import json
import time
import argparse
import torch
from common import ROOT, load_features, save_json, scores
from analyze_results import verify_ecg
from algorithm_config import selection_metric, validate_contract, PAPER_SUFFIX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default=str(ROOT / ('runs/maven_direction_full_incoming_paper' + PAPER_SUFFIX)))
    from pathlib import Path
    directory = Path(parser.parse_args().directory)
    folder = directory / 'fold_1'
    config = json.loads((directory / 'configuration.json').read_text(encoding='utf-8'))
    validate_contract(config)
    assert config['dataset'] == 'maven' and config['task'] == 'direction'
    assert config['epochs'] == 30 and config['limit_train'] == config['limit_eval'] == 0
    assert config['graph_orientation'] == 'incoming' and config['vocab_size'] == 30524
    checkpoint = folder / 'last.pt'
    version = checkpoint.stat().st_mtime_ns
    state = torch.load(checkpoint, map_location='cpu')
    history = [json.loads(line) for line in (folder / 'history.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [record['epoch'] for record in history] == list(range(1, state['epoch'] + 1))
    assert state['scheduler']['last_epoch'] == state['epoch'] * 2621
    best = json.loads((folder / 'best.json').read_text(encoding='utf-8'))
    assert state['best'] == best
    selected = max(history, key=lambda row: (row['validation'][selection_metric(config)]['all']['f1'], row['epoch']))
    assert selected['epoch'] == best['epoch']
    embedding = state['model']['pretrained_model.embeddings.word_embeddings.weight']
    assert list(embedding.shape) == [30524, 768]
    floating_dtypes = sorted({str(value.dtype) for value in state['model'].values() if value.is_floating_point()})
    assert floating_dtypes == ['torch.float32']
    assert all(key in state for key in ['random', 'numpy', 'torch', 'cuda'])
    predictions = json.loads((folder / 'best_predictions.json').read_text(encoding='utf-8'))
    test = load_features(ROOT / 'data/maven/test.pkl')
    assert len(predictions) == len(test) == 709
    gold, predicted, kinds = [], [], []
    for record, feature in zip(predictions, test):
        assert record['doc_id'] == str(feature.doc_id)
        assert record['gold'] == list(map(int, feature.target))
        assert record['rel_type'] == list(map(int, feature.rel_type))
        assert record['event_pairs'] == [list(map(int, pair)) for pair in feature.event_pairs]
        assert len(record['gold']) == len(record['pred']) == len(record['confidence'])
        verify_ecg(record)
        gold.extend(record['gold'])
        predicted.extend(record['pred'])
        kinds.extend(record['rel_type'])
    computed = scores(gold, predicted, kinds)
    assert computed == best['test']
    assert computed['direction']['all']['gold'] == 9698
    assert len(gold) == 260063
    assert checkpoint.stat().st_mtime_ns == version, 'Checkpoint advanced during audit; retry at a stable epoch boundary'
    result = dict(checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        checkpoint_epoch=state['epoch'], best_epoch=best['epoch'], expected_epochs=30,
        final_experiment=(folder / 'result.json').exists(), independent_model_reload_checked=False,
        test_documents=709, test_pairs=len(gold), test_positive_pairs=9698,
        predictions_match_canonical_data=True, saved_metrics_recomputed=True,
        causal_graphs_match=True, embedding_shape=list(embedding.shape),
        floating_dtypes=floating_dtypes, scheduler_steps=state['scheduler']['last_epoch'],
        optimizer_learning_rates=[group['lr'] for group in state['optimizer']['param_groups']],
        metrics=computed)
    save_json(ROOT / 'artifacts/maven_interim_verification.json', result)
    print({key: value for key, value in result.items() if key != 'metrics'})


if __name__ == '__main__':
    main()
