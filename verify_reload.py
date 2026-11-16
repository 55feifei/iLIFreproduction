"""Compare independently reloaded model outputs against saved fold outputs."""
import argparse
import json
from pathlib import Path
from common import ROOT, save_json
from ablation_preflight import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--prediction-file', required=True, type=Path)
    cli = parser.parse_args()
    checkpoint = cli.checkpoint.resolve()
    fold_path = checkpoint.parent
    reference = json.loads((fold_path / 'best_predictions.json').read_text(encoding='utf-8'))
    result = json.loads((fold_path / 'result.json').read_text(encoding='utf-8'))
    reloaded = json.loads(cli.prediction_file.read_text(encoding='utf-8'))
    assert Path(reloaded['checkpoint']).resolve() == checkpoint
    assert reloaded['split'] == 'test'
    assert len(reference) == len(reloaded['predictions'])
    confidence_difference = 0.
    for expected, actual in zip(reference, reloaded['predictions']):
        for field in ['doc_id', 'gold', 'pred', 'rel_type', 'event_pairs', 'final_ecg',
                      'argmax_pred', 'prediction_protocol', 'relation_threshold']:
            assert expected.get(field) == actual.get(field), (expected['doc_id'], field)
        assert len(expected['confidence']) == len(actual['confidence']) == len(expected['pred'])
        confidence_difference = max(confidence_difference, max(
            [abs(a - b) for a, b in zip(expected['confidence'], actual['confidence'])], default=0.))
    assert confidence_difference <= 1e-5
    assert result['test'] == reloaded['metrics']
    report = dict(checkpoint=str(checkpoint), prediction_file=str(cli.prediction_file.resolve()),
                  matched_documents=len(reference), best_epoch=result['epoch'],
                  max_confidence_difference=confidence_difference,
                  metrics_match=True, causal_graphs_match=True)
    report['sha256'] = {name: sha256(fold_path / name) for name in
                        ['best.pt', 'best_predictions.json', 'result.json', 'split.json']}
    report['configuration_sha256'] = sha256(fold_path.parent / 'configuration.json')
    report['prediction_sha256'] = sha256(cli.prediction_file)
    save_json(fold_path / 'reload_verification.json', report)
    save_json(ROOT / ('artifacts/reload_verification_%s_%s.json' % (fold_path.parent.name, fold_path.name)), report)
    print(report)


if __name__ == '__main__':
    main()
