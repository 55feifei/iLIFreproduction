import hashlib
import json
from common import ROOT, describe, load_features, save_json


def main():
    directory = ROOT / 'data/maven'
    report = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    for split in ['train', 'dev', 'test']:
        path = directory / (split + '.pkl')
        data = load_features(path)
        report[split].update(describe(data))
        report[split]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    report['complete_trigger_window_boundary_fixed'] = True
    save_json(directory / 'manifest.json', report)
    sources = {}
    for directory in [ROOT.parent / 'reference_source', ROOT / 'model']:
        for path in sorted(directory.rglob('*.py')):
            sources[str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(ROOT / 'artifacts/source_hashes.json', sources)


if __name__ == '__main__':
    main()
