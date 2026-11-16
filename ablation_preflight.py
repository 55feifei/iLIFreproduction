"""Check server inputs before starting Table 3; safe without torch installed."""
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path

PACKAGES = ('torch', 'transformers', 'torch_geometric', 'torch_sparse',
            'torch_scatter', 'numpy', 'scipy', 'sklearn', 'matplotlib')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check(model_path, esc_data, tasks):
    report = dict(python=sys.executable, python_version=platform.python_version(),
                  model_path=str(model_path), esc_data=str(esc_data),
                  packages={}, files={}, errors=[], warnings=[])
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            report['packages'][name] = getattr(module, '__version__', 'unknown')
        except Exception as exc:
            report['errors'].append('%s: %s: %s' % (name, type(exc).__name__, exc))
    if 'torch' in report['packages']:
        import torch
        report['cuda_available'] = torch.cuda.is_available()
        report['torch_cuda'] = torch.version.cuda
        if not report['cuda_available']:
            report['errors'].append('CUDA is required for benchmark training')
        else:
            report['gpu'] = torch.cuda.get_device_name(0)
        if not torch.__version__.startswith('1.11.0'):
            report['warnings'].append('Reference environment uses torch 1.11.0; validate any version change')
    if report['packages'].get('transformers') not in (None, '4.21.1'):
        report['warnings'].append('Training uses transformers.AdamW; use transformers 4.21.1 as pinned')
    required = [Path(model_path) / 'config.json', Path(model_path) / 'pytorch_model.bin']
    for task in tasks:
        directory = Path(esc_data) / 'exist' if task == 'existence' else Path(esc_data)
        required += [directory / ('EventStoryLine_bert_intra_and_inter_%s_features.pkl' % split)
                     for split in ('train', 'dev')]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            report['errors'].append('Missing or empty input: ' + str(path))
        else:
            report['files'][str(path)] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    report['status'] = 'passed' if not report['errors'] else 'blocked'
    manifest = Path(esc_data) / 'exist/derived_manifest.json'
    if 'existence' in tasks and manifest.is_file():
        report['existence_cache_provenance'] = json.loads(manifest.read_text(encoding='utf-8'))
    return report
