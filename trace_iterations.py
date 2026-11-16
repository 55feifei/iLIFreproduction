"""Measure actual initial/final predictions and ECGs from a saved model.

This diagnoses Figure 5's mechanism; it is not the retrained Table 3 ablation.
"""
import argparse
import json
from pathlib import Path
import torch
from common import ROOT, ESC_DATA, batch, load_features, scores, save_json
from algorithm_config import build_model, profile, PAPER_PROFILES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    cli = parser.parse_args()
    checkpoint = cli.checkpoint.resolve()
    config = json.loads((checkpoint.parent.parent / 'configuration.json').read_text(encoding='utf-8'))
    dataset, task = config['dataset'], config['task']
    device = torch.device('cuda')
    torch.set_num_threads(4)
    model = build_model(config, device)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()
    model.record_trace = True
    if dataset == 'esc':
        directory = ESC_DATA / 'exist' if task == 'existence' else ESC_DATA
        cache_split = 'dev' if cli.split == 'dev' else 'train'
        features = load_features(directory / ('EventStoryLine_bert_intra_and_inter_%s_features.pkl' % cache_split))
        if cli.split == 'test':
            ids = set(json.loads((checkpoint.parent / 'split.json').read_text(encoding='utf-8'))['test'])
            features = [f for f in features if str(f.doc_id) in ids]
    else:
        features = load_features(ROOT / ('data/maven/%s.pkl' % cli.split))
    ids = set(json.loads((checkpoint.parent / 'split.json').read_text(encoding='utf-8'))[cli.split])
    features = [f for f in features if str(f.doc_id) in ids]
    gold, types = [], []
    predictions = {key: [] for key in ['initial_argmax', 'final_argmax', 'initial_ecg', 'final_ecg']}
    documents = []
    with torch.no_grad():
        for f in features:
            _, logits = model(**batch(f, device))
            assert torch.equal(logits.cpu(), model.iteration_trace[-1])
            frames = []
            for frame in model.iteration_trace:
                confidence, label = frame.softmax(-1).max(-1)
                rejected = confidence < .6 if profile(config) in PAPER_PROFILES else confidence <= .6
                gated = label.masked_fill(rejected, 0)
                frames.append(dict(pred=label.tolist(), ecg_pred=gated.tolist()))
            gold.extend(map(int, f.target))
            types.extend(map(int, f.rel_type))
            for key, values in [('initial_argmax', frames[0]['pred']), ('final_argmax', frames[-1]['pred']),
                                ('initial_ecg', frames[0]['ecg_pred']), ('final_ecg', frames[-1]['ecg_pred'])]:
                predictions[key].extend(values)
            documents.append(dict(doc_id=str(f.doc_id), iterations=len(frames),
                gold=list(map(int, f.target)), rel_type=list(map(int, f.rel_type)),
                event_pairs=[list(map(int, pair)) for pair in f.event_pairs], frames=frames))
    result = dict(checkpoint=str(checkpoint), dataset=dataset, task=task, split=cli.split,
        checkpoint_fold_complete=(checkpoint.parent / 'result.json').exists(),
        partial_experiment=config['partial_experiment'],
        metrics={key: scores(gold, values, types) for key, values in predictions.items()},
        documents=documents)
    result['ecg_direction_metrics_applicable'] = config.get('ablation', 'none') != 'no_direction'
    if not result['ecg_direction_metrics_applicable']:
        for key in ['initial_ecg', 'final_ecg']:
            result['metrics'][key] = {'existence': result['metrics'][key]['existence']}
    save_json(cli.output, result)
    print(json.dumps(dict(documents=len(documents), min_iterations=min(d['iterations'] for d in documents),
        max_iterations=max(d['iterations'] for d in documents),
        standard_direction_f1={key: value.get('direction', {}).get('all', {}).get('f1')
            for key, value in result['metrics'].items()}), indent=2))


if __name__ == '__main__':
    main()
