"""MAVEN-ERE: original validation as test, 90/10 train/dev, 254/64 windows.

Retain the author's first mention per event and per-document half reversal.
Repair relation-type alignment after pair shuffling. Omit unused quadratic masks.
"""
import argparse
import importlib.util
import json
import math
import pickle
import random
from pathlib import Path
import torch
from transformers import BertTokenizer
from common import ROOT, AUTHOR, Features, describe, save_json


def convert(document, tokenizer):
    pieces, offsets = [], {}
    for sentence_id, words in enumerate(document.words):
        for word_id, word in enumerate(words):
            start = len(pieces)
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            if not token_ids:
                token_ids = [tokenizer.unk_token_id]
            pieces.extend(token_ids)
            offsets[(sentence_id, word_id)] = (start, len(pieces))
    spans = []
    for event in document.events:
        sentence, (start, end) = event['sent_id'], event['offset']
        span = (offsets[(sentence, start)][0], offsets[(sentence, end - 1)][1])
        expected = tokenizer.encode(event['trigger_word'], add_special_tokens=False)
        assert pieces[span[0]:span[1]] == expected, (document.id, event['trigger_word'])
        spans.append(span)
    starts = [0] if len(pieces) <= 254 else [64 * n for n in range(math.ceil((len(pieces) - 254) / 64) + 1)]
    windows = [torch.tensor([101] + pieces[start:start + 254] + [102]) for start in starts]
    ids = torch.nn.utils.rnn.pad_sequence(windows, batch_first=True, padding_value=0)
    ids = torch.nn.functional.pad(ids, (0, 256 - ids.shape[1]))
    nodes = []
    for start, end in spans:
        occurrences = [[i, [start - offset + 1, end - offset + 1]]
                       for i, offset in enumerate(starts) if offset <= start and end <= offset + 254]
        if not occurrences:
            raise ValueError('Trigger does not fit any window: ' + document.id)
        nodes.append(occurrences[0] if len(starts) == 1 else occurrences)
    feature = Features()
    feature.doc_id = document.id
    feature.enc_input_ids, feature.enc_mask_ids = ids, (ids != 0).long()
    feature.node_event, feature.event_pairs = nodes, document.event_pairs
    feature.t1_pos, feature.t2_pos, feature.target = document.t1_pos, document.t2_pos, document.labels
    feature.source_rel_type = document.rel_type
    feature.rel_type = [int(document.events[a]['sent_id'] != document.events[b]['sent_id'])
                        for a, b in document.event_pairs]
    feature.sentences = document.sentences
    return feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    random.seed(209)
    # Reuse source label and pair sampling logic, including RNG consumption.
    spec = importlib.util.spec_from_file_location('author_maven', AUTHOR / 'maven_code/preprocess_maven.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = BertTokenizer.from_pretrained(str(ROOT / 'models/bert-base-uncased'), local_files_only=True)
    directory = ROOT / 'data' / ('maven_pilot' if args.limit else 'maven')
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(seed=209, window=254, stride=64, first_mention_only=True,
                  relation_type_alignment_fixed=True, limit=args.limit)
    for filename in ['train', 'valid']:
        examples = []
        with (AUTHOR / ('data/maven_data/%s.jsonl' % filename)).open(encoding='utf-8') as stream:
            for line in stream:
                doc = module.Document(json.loads(line))
                if len(doc.events) >= 2:
                    examples.append(doc)
                if args.limit and len(examples) >= args.limit:
                    break
        if filename == 'train':
            train = random.sample(examples, int(len(examples) * .9))
            held = set(id(x) for x in train)
            splits = [('train', train), ('dev', [x for x in examples if id(x) not in held])]
        else:
            splits = [('test', examples)]
        for split, docs in splits:
            features = []
            for i, document in enumerate(docs, 1):
                features.append(convert(document, tokenizer))
                if i % 100 == 0 or i == len(docs):
                    print('%s %d/%d' % (split, i, len(docs)), flush=True)
            tmp = directory / (split + '.tmp')
            with tmp.open('wb') as stream:
                pickle.dump(features, stream, protocol=4)
            tmp.replace(directory / (split + '.pkl'))
            report[split] = describe(features)
            report[split]['source_rel_type_mismatches'] = sum(
                sum(a != b for a, b in zip(f.rel_type, f.source_rel_type)) for f in features)
    save_json(directory / 'manifest.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
