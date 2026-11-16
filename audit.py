import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
import torch
from common import ROOT, AUTHOR, ESC_DATA, describe, load_features, save_json


def main():
    report = dict(python=sys.version, executable=sys.executable,
                  cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  packages={})
    report['numeric_precision_defaults'] = dict(
        parameter_dtype=str(torch.get_default_dtype()),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    for name in ['torch', 'transformers', 'torch-geometric', 'torch-sparse',
                 'torch-scatter', 'numpy', 'scipy', 'scikit-learn', 'tqdm']:
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
    report['eventstoryline'] = {}
    for task, directory in [('direction', ESC_DATA), ('existence', ESC_DATA / 'exist')]:
        report['eventstoryline'][task] = {}
        for split in ['train', 'dev']:
            path = directory / ('EventStoryLine_bert_intra_and_inter_%s_features.pkl' % split)
            data = load_features(path)
            summary = describe(data)
            summary['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            summary['path'] = str(path)
            report['eventstoryline'][task][split] = summary
    report['maven_raw'] = {}
    for path in (AUTHOR / 'data/maven_data').glob('*.jsonl'):
        docs = [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
        relation_counts = Counter()
        for doc in docs:
            relation_counts.update({k: len(v) for k, v in doc.get('causal_relations', {}).items()})
        report['maven_raw'][path.name] = dict(documents=len(docs),
            labeled_documents=sum('causal_relations' in d for d in docs),
            causal_relations=dict(relation_counts),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save_json(ROOT / 'artifacts/audit.json', report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
