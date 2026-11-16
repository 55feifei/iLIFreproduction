import difflib
import hashlib
import json
from common import ROOT, PROJECT, ESC_DATA, load_features, save_json


def main():
    commit = json.loads((ROOT/'artifacts/official_publication_commit.json').read_text(encoding='utf-8-sig'))
    if isinstance(commit, list):
        commit = commit[0]
    tree = json.loads((ROOT/'artifacts/official_publication_tree.json').read_text(encoding='utf-8-sig'))
    remote = {entry['path']: entry for entry in tree['tree']}
    report = dict(commit=commit['sha'], date=commit['commit']['author']['date'], files={}, data={})
    diffs = []
    for name in ['model/CGE.py','model/model.py','parameter.py','utils.py','main.py','requirements.txt']:
        publication = ROOT/'official_publication_source'/name
        provided = PROJECT/'reference_source'/name
        a, b = publication.read_text(encoding='utf-8').splitlines(), provided.read_text(encoding='utf-8').splitlines()
        report['files'][name] = dict(publication_sha256=hashlib.sha256(publication.read_bytes()).hexdigest(),
            provided_sha256=hashlib.sha256(provided.read_bytes()).hexdigest(), equal_ignoring_line_endings=a==b)
        diffs.extend(difflib.unified_diff(a,b,fromfile='publication/'+name,tofile='provided/'+name,lineterm=''))
    for split in ['train','dev']:
        name = 'data/database/EventStoryLine_bert_intra_and_inter_%s_features.pkl' % split
        data = (PROJECT/'reference_source'/name).read_bytes()
        local_blob = hashlib.sha1(('blob %d\0' % len(data)).encode()+data).hexdigest()
        report['data'][split] = dict(local_git_blob_sha1=local_blob, publication_git_blob_sha1=remote[name]['sha'],
                                    matches=local_blob==remote[name]['sha'])
        direction = load_features(ESC_DATA/('EventStoryLine_bert_intra_and_inter_%s_features.pkl' % split))
        existence = load_features(ESC_DATA/('exist/EventStoryLine_bert_intra_and_inter_%s_features.pkl' % split))
        assert len(direction) == len(existence)
        changed_positive, changed_negative, sequences_changed = 0,0,0
        for d,e in zip(direction,existence):
            assert str(d.doc_id)==str(e.doc_id)
            d_pairs = [tuple(map(int,pair)) for pair in d.event_pairs]
            e_pairs = [tuple(map(int,pair)) for pair in e.event_pairs]
            d_map = {tuple(sorted(pair)): (pair,int(label)) for pair,label in zip(d_pairs,d.target)}
            assert set(d_map)==set(tuple(sorted(pair)) for pair in e_pairs)
            sequences_changed += d_pairs != e_pairs
            for pair,label in zip(e_pairs,e.target):
                original, original_label = d_map[tuple(sorted(pair))]
                assert int(label)==int(original_label>0)
                assert pair == (original[::-1] if original_label == 2 else original)
                if pair != original:
                    if original_label:
                        changed_positive += 1
                    else:
                        changed_negative += 1
        report['data'][split].update(existence_gold_by_unordered_pair_equal=True,
            documents_with_changed_pair_order=sequences_changed,
            reordered_positive_pairs=changed_positive,reordered_negative_pairs=changed_negative)
    (ROOT/'artifacts/publication_vs_provided.diff').write_text('\n'.join(diffs)+'\n',encoding='utf-8')
    save_json(ROOT/'artifacts/publication_version_audit.json',report)
    print(report)


if __name__ == '__main__':
    main()
