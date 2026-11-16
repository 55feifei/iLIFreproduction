"""Derive binary ESC inputs from direction caches using the paper Appendix B protocol.

CAUSE stays ordered; EFFECT pairs are reversed; both become positive label 1.
Only create missing caches. Record derivation explicitly, without presenting
these files as the author's original binary-cache bytes.
"""
import argparse
import copy
import json
from pathlib import Path
import pickle
import zipfile
from ablation_preflight import sha256
from project_paths import ROOT, esc_data_path

FILENAME = 'EventStoryLine_bert_intra_and_inter_%s_features.pkl'


def convert(feature):
    result = copy.copy(feature)
    pairs, labels = [], []
    if not (len(feature.event_pairs) == len(feature.target) == len(feature.rel_type)):
        raise ValueError('Event-pair / label / relation length mismatch')
    for pair, label in zip(feature.event_pairs, feature.target):
        label = int(label)
        if label not in (0, 1, 2):
            raise ValueError('Expected direction labels 0/1/2')
        source, target = map(int, pair)
        if label == 2:
            source, target = target, source
        pairs.append([source, target])
        labels.append(int(label > 0))
    result.event_pairs, result.target = pairs, labels
    result.t1_pos = [pair[0] for pair in pairs]
    result.t2_pos = [pair[1] for pair in pairs]
    return result


def validate(direction, existence):
    if len(direction) != len(existence):
        raise ValueError('Direction/existence document counts differ')
    reversals = 0
    for original, actual in zip(direction, existence):
        expected = convert(original)
        if str(original.doc_id) != str(actual.doc_id):
            raise ValueError('Document order differs')
        for field in ('event_pairs', 'target', 't1_pos', 't2_pos', 'rel_type'):
            # Caches can store lists, tuples or NumPy integer scalars.
            if list(getattr(actual, field)) != list(getattr(expected, field)):
                raise ValueError('Binary cache differs from derived protocol: ' + field)
        if not original.enc_input_ids.equal(actual.enc_input_ids) or not original.enc_mask_ids.equal(actual.enc_mask_ids):
            raise ValueError('Token inputs differ')
        if original.node_event != actual.node_event or original.sentences != actual.sentences:
            raise ValueError('Event spans or sentences differ')
        reversals += sum(int(label) == 2 for label in original.target)
    return dict(documents=len(direction), reversed_effect_pairs=reversals)


def prepare(directory):
    from common import load_features, save_json
    import torch
    inputs = {split: directory / (FILENAME % split) for split in ('train', 'dev')}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing direction cache: ' + ', '.join(missing))
    manifest = dict(protocol='esc_existence_derived_from_direction_v1',
        paper='https://aclanthology.org/2024.acl-long.210.pdf',
        origin='derived cache; not claimed byte-identical to author binary cache',
        positive_direction='CAUSE; reverse EFFECT event pairs, then collapse labels',
        pair_order='preserve source row order', torch_version=torch.__version__, splits={})
    loaded = {}
    for split, source in inputs.items():
        direction = load_features(source)
        loaded[split] = direction
        destination = directory / 'exist' / source.name
        created = not destination.exists()
        if created:
            existence = [convert(feature) for feature in direction]
            validate(direction, existence)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.pkl.tmp')
            with temporary.open('wb') as stream:
                pickle.dump(existence, stream, protocol=4)
            temporary.replace(destination)
        existence = load_features(destination)
        summary = validate(direction, existence)
        summary.update(direction_sha256=sha256(source), existence_sha256=sha256(destination),
                       created=created)
        manifest['splits'][split] = summary
    train_ids = [str(feature.doc_id) for feature in loaded['train']]
    dev_ids = [str(feature.doc_id) for feature in loaded['dev']]
    if len(set(train_ids)) != len(train_ids) or len(set(dev_ids)) != len(dev_ids) or set(train_ids) & set(dev_ids):
        raise ValueError('Duplicate documents or train/dev overlap')
    manifest_path = directory / 'exist/derived_manifest.json'
    if not manifest_path.exists():
        save_json(manifest_path, manifest)
    else:
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        for split in ('train', 'dev'):
            for key in ('direction_sha256', 'existence_sha256'):
                if existing['splits'][split][key] != manifest['splits'][split][key]:
                    raise ValueError('Derived input provenance changed: ' + split)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--esc-data', type=Path, default=esc_data_path())
    parser.add_argument('--archive', type=Path, help='Optionally build an ESC-only zip for upload')
    cli = parser.parse_args()
    directory = cli.esc_data.expanduser().resolve()
    manifest = prepare(directory)
    if cli.archive:
        archive = cli.archive.expanduser().resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for split in ('train', 'dev'):
                for prefix in ('', 'exist/'):
                    name = prefix + FILENAME % split
                    bundle.write(directory / name, 'esc/' + name)
            bundle.write(directory / 'exist/derived_manifest.json', 'esc/exist/derived_manifest.json')
        temporary.replace(archive)
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise ValueError('Archive integrity check failed')
        manifest['archive'] = str(archive)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
