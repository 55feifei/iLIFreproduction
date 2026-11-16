"""Load a saved best model and export predictions/ECGs for an existing split."""
import argparse
import json
import torch
from common import ROOT, ESC_DATA, load_features, save_json
from run import evaluate
from legacy_utils import setup_seed
from algorithm_config import build_model
from project_paths import esc_data_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--model-path', type=str, help='Override a relocated local BERT directory')
    parser.add_argument('--esc-data', type=str, help='Override relocated ESC cache directory')
    cli = parser.parse_args()
    torch.set_num_threads(4)
    setup_seed(209)
    from pathlib import Path
    checkpoint = Path(cli.checkpoint).resolve()
    config = json.loads((checkpoint.parent.parent / 'configuration.json').read_text(encoding='utf-8'))
    if cli.model_path:
        config['model_path'] = cli.model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config, device)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    if config['dataset'] == 'maven':
        features = load_features(ROOT / ('data/maven/%s.pkl' % cli.split))
    else:
        base = esc_data_path(cli.esc_data or config.get('esc_data'))
        directory = base / 'exist' if config['task'] == 'existence' else base
        split = 'dev' if cli.split == 'dev' else 'train'
        features = load_features(directory / ('EventStoryLine_bert_intra_and_inter_%s_features.pkl' % split))
        if cli.split == 'test':
            ids = set(json.loads((checkpoint.parent / 'split.json').read_text(encoding='utf-8'))['test'])
            features = [f for f in features if str(f.doc_id) in ids]
    # Reconstruct the saved subset for pilots as well as full experiments.
    ids = set(json.loads((checkpoint.parent / 'split.json').read_text(encoding='utf-8'))[cli.split])
    features = [f for f in features if str(f.doc_id) in ids]
    if cli.limit:
        features = features[:cli.limit]
    metrics, predictions = evaluate(model, features, device)
    save_json(cli.output, dict(checkpoint=str(checkpoint), split=cli.split,
                             metrics=metrics, predictions=predictions))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
