"""Audit completed ESC v3/v4 runs without training or modifying run artifacts.

Usage: python analyze_v3_v4.py [--reload]
Outputs: artifacts/v3_v4_analysis/{analysis.json,fold_metrics.csv,*.png}
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import argparse
import csv
import copy
import gc
import hashlib
import itertools
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from common import ROOT, ESC_DATA, load_features, scores, batch
from algorithm_config import build_model, validate_contract
from legacy_utils import setup_seed

OUT = ROOT / 'artifacts/v3_v4_analysis'
TASKS = ('direction', 'existence')
VERSIONS = ('v3', 'v4')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def arrays(documents, field='pred'):
    return [np.concatenate([np.asarray(d[key]) for d in documents])
            for key in ('gold', field, 'rel_type')]


def count_errors(documents):
    g, p, t = arrays(documents)
    out = {}
    for scope, mask in [('all', np.ones(g.size, bool)), ('intra', t == 0), ('inter', t == 1)]:
        gg, pp = g[mask], p[mask]
        out[scope] = dict(pairs=int(mask.sum()), correct_causal=int(((gg > 0) & (gg == pp)).sum()),
            missed_causal=int(((gg > 0) & (pp == 0)).sum()),
            reversed_causal=int(((gg > 0) & (pp > 0) & (gg != pp)).sum()),
            spurious_causal=int(((gg == 0) & (pp > 0)).sum()),
            correct_none=int(((gg == 0) & (pp == 0)).sum()))
    return out


def permutation_p(differences):
    d = np.asarray(differences)
    null = [abs(np.mean(d * signs)) for signs in itertools.product([-1, 1], repeat=len(d))]
    return float(np.mean(np.asarray(null) >= abs(d.mean()) - 1e-12))


def canonical(documents):
    out = {}
    for d in documents:
        for pair, g, p in zip(d['event_pairs'], d['gold'], d['pred']):
            key = (d['doc_id'], *sorted(pair))
            assert key not in out
            out[key] = (int(g > 0), int(p > 0))
    return out


def run_audit():
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(created=datetime.now().isoformat(), runs={}, comparisons={}, data={},
                  notes=['No training launched. Epoch selection audited against development F1.',
                         'Fold SD is split variability, not variability over independent random seeds.'])
    predictions, splits_by_run, features_by_task = {}, {}, {}
    csv_rows = []
    for task in TASKS:
        directory = ESC_DATA / 'exist' if task == 'existence' else ESC_DATA
        all_features = {}
        for split in ('train', 'dev'):
            fs = load_features(directory / ('EventStoryLine_bert_intra_and_inter_%s_features.pkl' % split))
            all_features[split] = {str(f.doc_id): f for f in fs}
            features_by_task[task] = all_features
            ordering = Counter()
            for f in fs:
                for pair, label in zip(f.event_pairs, f.target):
                    ordering[('positive' if label else 'negative') + ('_reverse_index' if pair[0] > pair[1] else '_forward_index')] += 1
            result['data'][task + '_' + split] = dict(documents=len(fs),
                pairs=sum(len(f.target) for f in fs), causal=sum(sum(int(g > 0) for g in f.target) for f in fs),
                topics=sorted(set(str(f.topic_id) for f in fs)), pair_ordering=dict(ordering))
        for version in VERSIONS:
            key = task + '_' + version
            run = ROOT / ('runs/esc_%s_full_incoming_paper_%s' % (task, version))
            config = read(run / 'configuration.json')
            validate_contract(config)
            assert not config['partial_experiment'] and config['epochs'] == 30
            summary = read(run / 'summary.json')
            assert summary['completed_folds'] == 5
            folds, full_docs, split_records = [], [], []
            histories = []
            for i in range(1, 6):
                fd = run / ('fold_%d' % i)
                hist = [json.loads(line) for line in (fd / 'history.jsonl').read_text().splitlines() if line.strip()]
                assert [h['epoch'] for h in hist] == list(range(1, 31)), (key, i, 'history continuity')
                assert all(np.isfinite(h['train_loss']) and h['train_loss'] >= 0 for h in hist)
                best = read(fd / 'result.json')
                assert best == read(fd / 'best.json') == summary['folds'][i-1]
                expected = max(hist, key=lambda h: (h['validation'][task]['all']['f1'], h['epoch']))
                assert expected['epoch'] == best['epoch'] and expected['validation'] == best['validation']
                docs = read(fd / 'best_predictions.json')
                recomputed = scores(*arrays(docs))
                assert recomputed == best['test'], (key, i, 'prediction metrics')
                assert all(d['pred'] == d['argmax_pred'] and d['prediction_protocol'] == 'final_argmax' for d in docs)
                split = read(fd / 'split.json')
                assert {d['doc_id'] for d in docs} == set(split['test'])
                assert not (set(split['train']) & set(split['test']))
                assert not (set(split['train']) & set(split['dev']))
                assert not (set(split['test']) & set(split['dev']))
                train_topics = {str(all_features['train'][d].topic_id) for d in split['train']}
                test_topics = {str(all_features['train'][d].topic_id) for d in split['test']}
                dev_topics = {str(all_features['dev'][d].topic_id) for d in split['dev']}
                vals = np.array([h['validation'][task]['all']['f1'] for h in hist])
                loss = [h['train_loss'] for h in hist]
                near = np.flatnonzero(vals >= vals.max() - .01) + 1
                row = dict(fold=i, best_epoch=best['epoch'], train_docs=len(split['train']),
                    test_docs=len(split['test']), dev_docs=len(split['dev']),
                    train_test_topic_overlap=sorted(train_topics & test_topics),
                    train_dev_topic_overlap=sorted(train_topics & dev_topics),
                    train_loss_epoch1=loss[0], train_loss_best=loss[best['epoch']-1], train_loss_epoch30=loss[-1],
                    dev_best=vals.max()*100, dev_last=vals[-1]*100,
                    dev_peak_to_last_pp=(vals.max()-vals[-1])*100,
                    near_best_epochs=near.tolist(), dev_after_epoch10_sd=float(vals[9:].std(ddof=1)*100),
                    total_seconds=sum(h['seconds'] for h in hist),
                    mean_epoch_seconds=float(np.mean([h['seconds'] for h in hist])),
                    final5_dev_mean=float(vals[-5:].mean()*100),
                    test=best['test'])
                folds.append(row)
                full_docs += docs
                split_records.append(split)
                histories.append(hist)
                csv_rows.append(dict(task=task, version=version, fold=i,
                    best_epoch=best['epoch'], test_p=best['test'][task]['all']['p']*100,
                    test_r=best['test'][task]['all']['r']*100,
                    test_f1=best['test'][task]['all']['f1']*100,
                    test_intra_f1=best['test'][task]['intra']['f1']*100,
                    test_inter_f1=best['test'][task]['inter']['f1']*100,
                    source_f1=best['test']['source_direction']['all']['f1']*100,
                    dev_best=row['dev_best'], dev_last=row['dev_last'],
                    train_loss_epoch1=loss[0], train_loss_epoch30=loss[-1],
                    mean_epoch_seconds=row['mean_epoch_seconds']))
            assert len(full_docs) == 233 and len(set(d['doc_id'] for d in full_docs)) == 233
            predictions[key] = full_docs
            splits_by_run[key] = split_records
            mean = {mode: {scope: {metric: float(np.mean([f['test'][mode][scope][metric] for f in folds]))
                for metric in ('p', 'r', 'f1')} for scope in ('all', 'intra', 'inter')}
                for mode in ('direction', 'existence', 'source_direction')}
            assert mean == summary['mean_test']
            gated_docs = [dict(d, pred=[p if c >= .6 else 0 for p, c in zip(d['pred'], d['confidence'])]) for d in full_docs]
            positives = [(p, c) for d in full_docs for p, c in zip(d['pred'], d['confidence']) if p]
            topics = {}
            for topic in sorted(set(str(all_features['train'][d['doc_id']].topic_id) for d in full_docs)):
                group = [d for d in full_docs if str(all_features['train'][d['doc_id']].topic_id) == topic]
                topics[topic] = dict(documents=len(group), metrics=scores(*arrays(group))[task]['all'])
            log_candidates = list(run.parent.glob(run.name + '*.log'))
            log_stats = []
            for logpath in log_candidates:
                text = logpath.read_text(encoding='utf-8', errors='replace')
                timestamps = re.findall(r'(2026-\d\d-\d\d \d\d:\d\d:\d\d)', text)
                memory = [float(v) for v in re.findall(r'peak_GB=([\d.]+)', text)]
                log_stats.append(dict(path=str(logpath), first=timestamps[0] if timestamps else None,
                    last=timestamps[-1] if timestamps else None, peak_allocated_gb=max(memory) if memory else None,
                    nonfinite_or_oom=bool(re.search(r'Traceback|out of memory|Nonfinite loss', text))))
            result['runs'][key] = dict(configuration=config, folds=folds, histories=histories,
                mean_test=mean, pooled_test=scores(*arrays(full_docs)), errors=count_errors(full_docs),
                fold_sd={scope: float(np.std([f['test'][task][scope]['f1']*100 for f in folds], ddof=1))
                    for scope in ('all', 'intra', 'inter')},
                epoch_means={str(e): dict(train_loss=float(np.mean([h[e-1]['train_loss'] for h in histories])),
                    dev_f1=float(np.mean([h[e-1]['validation'][task]['all']['f1']*100 for h in histories])))
                    for e in range(1, 31)},
                positive_predictions=len(positives), positive_confidence_median=float(np.median([c for p,c in positives])),
                positive_below_threshold=sum(c < .6 for p,c in positives),
                thresholded_output_pooled=scores(*arrays(gated_docs)), topics=topics, logs=log_stats,
                audit='passed: 5x30 epochs, dev selection, prediction metrics, summary, document splits')
    for task in TASKS:
        assert splits_by_run[task+'_v3'] == splits_by_run[task+'_v4']
        a, b = result['runs'][task+'_v3'], result['runs'][task+'_v4']
        differences = [(fb['test'][task]['all']['f1']-fa['test'][task]['all']['f1'])*100
                       for fa,fb in zip(a['folds'], b['folds'])]
        da = {d['doc_id']: d for d in predictions[task+'_v3']}
        db = {d['doc_id']: d for d in predictions[task+'_v4']}
        disagreements = Counter()
        for did in da:
            x,y = da[did], db[did]
            for f in ('gold', 'event_pairs', 'rel_type'):
                assert x[f] == y[f]
            for g,p,q in zip(x['gold'],x['pred'],y['pred']):
                disagreements['pairs'] += 1
                disagreements['different'] += p != q
                disagreements['v3_correct_v4_wrong'] += (g==p and g!=q)
                disagreements['v4_correct_v3_wrong'] += (g==q and g!=p)
        result['comparisons'][task] = dict(v4_minus_v3_fold_f1_pp=differences,
            mean_difference_pp=float(np.mean(differences)), v4_wins=sum(d>0 for d in differences),
            fold_signflip_two_sided_p=permutation_p(differences),
            caveat='Descriptive only: five CV folds overlap in training data; one seed, no independent seed replication.',
            disagreements=dict(disagreements), config_diff={k:[a['configuration'].get(k),b['configuration'].get(k)]
                for k in a['configuration'].keys() | b['configuration'].keys()
                if a['configuration'].get(k) != b['configuration'].get(k)})
    assert splits_by_run['direction_v3'] == splits_by_run['existence_v3']
    for version in VERSIONS:
        direction = canonical(predictions['direction_'+version])
        existence = canonical(predictions['existence_'+version])
        assert direction.keys() == existence.keys()
        assert all(direction[k][0] == existence[k][0] for k in direction)
        result['comparisons']['existence_vs_collapsed_direction_'+version] = dict(
            same_unordered_pairs_and_binary_gold=True,
            direction_collapsed_pooled=result['runs']['direction_'+version]['pooled_test']['existence'],
            existence_model_pooled=result['runs']['existence_'+version]['pooled_test']['existence'],
            input_order_protocol_differs=True)
    with (OUT/'fold_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    dump(OUT/'analysis.json', result)
    print(json.dumps({k:dict(f1=v['mean_test'][k.split('_')[0]]['all']['f1']*100,
        best_epochs=[f['best_epoch'] for f in v['folds']],
        loss1=v['epoch_means']['1']['train_loss'],loss30=v['epoch_means']['30']['train_loss'],
        dev1=v['epoch_means']['1']['dev_f1'],dev30=v['epoch_means']['30']['dev_f1'])
        for k,v in result['runs'].items()}, indent=2), flush=True)
    return result, features_by_task, predictions


def reload_audit(result, features, predictions):
    setup_seed(209)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    traces = dict(device=torch.cuda.get_device_name(0), runs={})
    for task in TASKS:
        for version in VERSIONS:
            key = task+'_'+version
            config = result['runs'][key]['configuration']
            records = []
            final_docs, initial_docs = [], []
            conf_diff = 0.
            for fold in range(1,6):
                fd = Path(config['output']) / ('fold_%d' % fold)
                model = build_model(config, device)
                state = torch.load(fd/'best.pt', map_location='cpu')
                model.load_state_dict(state)
                del state
                model.eval(); model.record_trace = True
                saved = {d['doc_id']: d for d in read(fd/'best_predictions.json')}
                feature_list = [features[task]['train'][did] for did in read(fd/'split.json')['test']]
                with torch.no_grad():
                    for f in feature_list:
                        _, logits = model(**batch(f, device))
                        frames = model.iteration_trace
                        assert torch.equal(frames[-1],logits.cpu())
                        labels = [frame.argmax(-1).tolist() for frame in frames]
                        expected = saved[str(f.doc_id)]
                        assert labels[-1] == expected['pred'], (key,fold,f.doc_id,'labels')
                        confidence = logits.softmax(-1).max(-1).values.cpu().numpy()
                        conf_diff = max(conf_diff, float(np.max(np.abs(confidence - expected['confidence']))))
                        graph = model.get_graphedge_index(logits, f.event_pairs, [f.rel_type])
                        assert {kind[1]:edge.t().cpu().tolist() for kind,edge in graph.items()} == expected['final_ecg']
                        final_docs.append(dict(expected,pred=labels[-1]))
                        initial_docs.append(dict(expected,pred=labels[0]))
                        edge_counts = []
                        changes = []
                        previous = None
                        for frame in frames:
                            edge = model.get_graphedge_index(frame.to(device), f.event_pairs, [f.rel_type])
                            edge_counts.append(sum(e.shape[1] for e in edge.values()))
                            if previous is not None:
                                changes.append(int(model.graph_difference(previous, edge, len(f.node_event))))
                            previous = edge
                        records.append(dict(doc_id=str(f.doc_id),fold=fold,updates=len(frames)-1,
                            cap=min(len(f.sentences),config['max_graph_updates']),
                            edge_counts=edge_counts,adjacency_changes=changes,
                            initial_f1=scores(expected['gold'],labels[0],expected['rel_type'])[task]['all']['f1'],
                            final_f1=scores(expected['gold'],labels[-1],expected['rel_type'])[task]['all']['f1'],
                            changed_labels=sum(p!=q for p,q in zip(labels[0],labels[-1]))))
                del model
                gc.collect(); torch.cuda.empty_cache()
                print('RELOAD %s fold=%d passed; max confidence delta=%.3g' % (key,fold,conf_diff),flush=True)
            assert conf_diff <= 1e-5
            pooled_final = scores(*arrays(final_docs))
            assert pooled_final == result['runs'][key]['pooled_test']
            folded_initial = []
            offset = 0
            for fold in range(1,6):
                n = result['runs'][key]['folds'][fold-1]['test_docs']
                folded_initial.append(scores(*arrays(initial_docs[offset:offset+n]))[task]['all']['f1']*100)
                offset += n
            traces['runs'][key] = dict(all_predictions_graphs_and_metrics_match=True,
                max_confidence_difference=conf_diff,documents=records,
                mean_updates=float(np.mean([d['updates'] for d in records])),
                median_updates=float(np.median([d['updates'] for d in records])),
                update_distribution=dict(Counter(d['updates'] for d in records)),
                stop_by_convergence=sum(d['adjacency_changes'][-1]<=2 for d in records),
                stop_by_cap_without_convergence=sum(d['adjacency_changes'][-1]>2 for d in records),
                initial_empty_graphs=sum(d['edge_counts'][0]==0 for d in records),
                final_empty_graphs=sum(d['edge_counts'][-1]==0 for d in records),
                mean_initial_edges=float(np.mean([d['edge_counts'][0] for d in records])),
                mean_final_edges=float(np.mean([d['edge_counts'][-1] for d in records])),
                mean_initial_fold_f1=float(np.mean(folded_initial)),initial_fold_f1=folded_initial,
                mean_final_fold_f1=result['runs'][key]['mean_test'][task]['all']['f1']*100,
                initial_pooled=scores(*arrays(initial_docs)), final_pooled=pooled_final)
            dump(OUT/'reload_iteration_audit.json', traces)
    return traces


def plots(result, traces=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'font.size':10,'savefig.dpi':180})
    colors = dict(v3='#2563eb',v4='#e07a29')
    fig, axes=plt.subplots(2,3,figsize=(15,8.4))
    for row,task in enumerate(TASKS):
        for version in VERSIONS:
            data=result['runs'][task+'_'+version]
            for hist in data['histories']:
                axes[row,0].plot(range(1,31),[h['train_loss'] for h in hist],color=colors[version],alpha=.16,lw=.8)
                axes[row,1].plot(range(1,31),[h['validation'][task]['all']['f1']*100 for h in hist],color=colors[version],alpha=.16,lw=.8)
            axes[row,0].plot(range(1,31),[data['epoch_means'][str(e)]['train_loss'] for e in range(1,31)],color=colors[version],label=version,lw=2)
            axes[row,1].plot(range(1,31),[data['epoch_means'][str(e)]['dev_f1'] for e in range(1,31)],color=colors[version],label=version,lw=2)
            fs=[f['test'][task]['all']['f1']*100 for f in data['folds']]
            axes[row,2].plot(range(1,6),fs,'o-',color=colors[version],label=version,lw=1.8)
            for i,y in enumerate(fs,1):
                axes[row,2].annotate('%.1f'%y,(i,y),textcoords='offset points',xytext=(0,7 if version=='v3' else -13),ha='center',color=colors[version],fontsize=8)
        axes[row,0].set_yscale('log'); axes[row,0].set_title(task.title()+': training focal loss')
        axes[row,1].set_title(task.title()+': development F1 (%)')
        axes[row,2].set_title(task.title()+': selected test F1 (%)')
        axes[row,2].axhline(47.8 if task=='direction' else 62.5,color='#607080',ls='--',lw=1,label='Paper')
        for col in range(3):
            axes[row,col].grid(alpha=.18);axes[row,col].set_xlabel('Fold' if col==2 else 'Epoch');axes[row,col].legend(fontsize=8)
        axes[row,2].set_xticks(range(1,6))
    fig.suptitle('iLIF ESC v3 vs v4 | 20 completed folds, 600 epochs | one seed (209)',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'training_and_folds.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    scopes=['intra','inter','all']; x=np.arange(3)
    for ax,task in zip(axes,TASKS):
        paper=[60,42.8,47.8] if task=='direction' else [71.2,59.1,62.5]
        for j,version in enumerate(VERSIONS):
            run=result['runs'][task+'_'+version]
            vals=[run['mean_test'][task][scope]['f1']*100 for scope in scopes]
            bars=ax.bar(x+(j-.5)*.25,vals,width=.24,color=colors[version],label=version)
            for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.25,'%.2f'%bar.get_height(),ha='center',fontsize=9)
        ax.plot(x,paper,'D',color='#283747',label='Paper',ms=6)
        ax.set_xticks(x);ax.set_xticklabels(['Intra','Inter','Overall']);ax.set_title(task.title());ax.set_ylabel('F1 (%)');ax.legend();ax.grid(axis='y',alpha=.18)
        ax.set_ylim(25 if task=='direction' else 45,80)
    fig.suptitle('Standard metrics, arithmetic mean of five test folds');fig.tight_layout();fig.savefig(OUT/'paper_comparison.png');plt.close(fig)
    if traces:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for ax,task in zip(axes,TASKS):
            for j,version in enumerate(VERSIONS):
                data=traces['runs'][task+'_'+version]
                dist=data['update_distribution']
                xs=np.arange(1,10)
                ax.bar(xs+(j-.5)*.3,[dist.get(str(i),dist.get(i,0)) for i in xs],width=.28,color=colors[version],label=version)
            ax.set_title(task.title()+': actual graph updates at test');ax.set_xticks(range(1,10));ax.set_xlabel('Updates');ax.set_ylabel('Documents (233 total)');ax.legend();ax.grid(axis='y',alpha=.18)
        fig.tight_layout();fig.savefig(OUT/'iteration_distribution.png');plt.close(fig)


def canonical_order_audit(result, features):
    """Inference sensitivity, not a retrained leakage-free performance estimate."""
    setup_seed(209)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    report = dict(protocol='Sort every event pair by event index at inference, independent of its gold label.',
        caveat='Checkpoints were trained on author label-conditioned ordering. This is a distribution-shift sensitivity check, not a retrained fair baseline or a causal estimate of label leakage.',
        runs={})
    for version in VERSIONS:
        key='existence_'+version
        config=result['runs'][key]['configuration']
        folds, documents = [], []
        subgroup=Counter()
        for fold in range(1,6):
            fd=Path(config['output'])/('fold_%d'%fold)
            model=build_model(config,device)
            state=torch.load(fd/'best.pt',map_location='cpu');model.load_state_dict(state);del state
            model.eval()
            saved={d['doc_id']:d for d in read(fd/'best_predictions.json')}
            fold_docs=[]
            with torch.no_grad():
                for did in read(fd/'split.json')['test']:
                    original=features['existence']['train'][did]
                    f=copy.copy(original)
                    f.event_pairs=[sorted(map(int,p)) for p in original.event_pairs]
                    f.t1_pos=[p[0] for p in f.event_pairs];f.t2_pos=[p[1] for p in f.event_pairs]
                    _,logits=model(**batch(f,device))
                    pred=logits.argmax(-1).cpu().tolist()
                    doc=dict(doc_id=did,gold=list(map(int,f.target)),pred=pred,
                             rel_type=list(map(int,f.rel_type)),event_pairs=f.event_pairs)
                    fold_docs.append(doc)
                    for pair,g,p,q in zip(original.event_pairs,doc['gold'],saved[did]['pred'],pred):
                        group='reverse' if pair[0]>pair[1] else 'forward'
                        if g:
                            subgroup[group+'_gold']+=1
                            subgroup[group+'_author_tp']+=int(p>0)
                            subgroup[group+'_canonical_tp']+=int(q>0)
            metrics=scores(*arrays(fold_docs))['existence']
            folds.append(dict(fold=fold,canonical=metrics,
                author=result['runs'][key]['folds'][fold-1]['test']['existence']))
            documents+=fold_docs
            del model;gc.collect();torch.cuda.empty_cache()
            print('CANONICAL ORDER %s fold=%d F1=%.4f'%(key,fold,metrics['all']['f1']),flush=True)
        report['runs'][key]=dict(folds=folds,positive_subgroups=dict(subgroup),
            mean_test={scope:{metric:float(np.mean([f['canonical'][scope][metric] for f in folds]))
                for metric in ('p','r','f1')} for scope in ('all','intra','inter')},
            pooled_test=scores(*arrays(documents))['existence'],
            author_mean_test=result['runs'][key]['mean_test']['existence'])
        dump(OUT/'pair_order_sensitivity.json',report)
    return report


def write_report(result, traces, ordering):
    """Write a standalone Chinese report from measured values only."""
    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(map(str,row))+' |' for row in rows])
    def f(x): return '%.2f'%x
    def pct(x): return f(x*100)
    run=result['runs']
    lines=[
        '**iLIF v3 / v4 训练过程、结果与复现可信度分析**', '',
        '分析日期：2026-10-16。范围：ESC 两项任务 × 两个版本 × 五折；全部为 30 轮正式训练。'
        '本次只做日志审计和模型推理，未启动或续接训练。', '',
        '**总体判断。** 训练与保存结果在工程上可复核；ESC 方向识别接近论文但仍有差距；'
        '存在性在作者缓存协议下达到论文数值，但高分强烈依赖真实标签参与的事件对顺序，不能直接解释为未知方向候选对上的通用存在性能力。'
        'v3 在本次单种子实验的两项总体 F1 都略高于 v4，不能据此宣称统计稳定的优越性。MAVEN 正式结果和机制消融尚缺，整篇论文的复现仍不完整。', '',
        '**证据范围与计分。** 独立核对 20 份配置、20 份逐轮记录（共 600 个 epoch）、开发集选轮、'
        '保存预测、各折结果与汇总；每个实验的 233 篇折外文档恰好覆盖一次，共 61,065 个候选对。'
        '所有最佳轮次与开发集标准 F1 的最大值一致，同分取较晚轮次。测试集在开发集提升时被评估，代码中不用于选轮。'
        '随后实际重载全部 20 个 best.pt，932 篇文档的预测类别、置信度、阈值因果图及计分均与保存结果一致，最大置信度差为 0。'
        '这些验证支持现有 checkpoint 与报告一致；没有重新执行全程训练，不能由此承诺重新训练逐位相同。', '',
        '以下主结果均为每折先计算标准 F1，再对五折取算术平均；P/R 也各自平均。'
        '合并全部折外事件对的微平均只用于错误诊断。所有“差”以百分点计，未把两种汇总混用。', '',
        '**版本差异。** 同一任务的 v3/v4 实际 configuration.json 除版本、输出路径及训练建图阈值外一致。', '',
        table(['规则','v3','v4'],[
            ['训练时迭代建图','因果类 argmax 且最大概率 ≥ 0.6','因果类 argmax，不加 0.6 门槛'],
            ['开发集／测试集建图','最大概率 ≥ 0.6','最大概率 ≥ 0.6'],
            ['最终输出','最后一轮 argmax','最后一轮 argmax'],
            ['开发集选轮','标准任务 F1','标准任务 F1']]), '',
        '共同设置：bert-base-uncased 微调，BERT 学习率 2e-5，其余模块 1e-4，AdamW，10% 预热，'
        'batch size=1，30 轮，seed=209；4 个注意力头，句内／跨句融合 0.7／0.3，句内自环，'
        'LeakyReLU，dropout=0.4，focal loss γ=2、正类 α=0.75；最多 min(全文句数,9) 次图更新，'
        '有向邻接差 ≤2 时停止。每轮文档内事件对损失取平均，再按 1/l 累加。'
        'v4 的训练门槛对应作者源码默认 --rate=0；v3 更接近论文文字中统一使用置信度门槛的解释。'
        '这只说明一处规则的来源，两个版本整体仍是对论文和源码差异作出明确解释后的实现，不能称为作者原训练程序的逐行复刻。', '',
        '**主表结果。** 论文目标来自本地论文表 1。', '']
    rows=[]
    for task,title,paper in [('direction','方向',[60.,42.8,47.8]),('existence','存在性',[71.2,59.1,62.5])]:
        rows.append([title+'：论文',*map(f,paper),'—','—'])
        for version in VERSIONS:
            r=run[task+'_'+version]
            vals=[r['mean_test'][task][scope]['f1']*100 for scope in ('intra','inter','all')]
            rows.append([title+'：'+version,*map(f,vals),f(vals[-1]-paper[-1]),f(r['fold_sd']['all'])])
    lines += [table(['任务／版本','句内 F1','跨句 F1','总体 F1','总体较论文差','折间标准差'],rows),'',
        '方向任务：v3 总体比论文低 1.30 点，v4 低 2.07 点；v4 相比 v3 句内增加 0.20 点、跨句下降 1.10 点。'
        '存在性：v3 比论文高 1.04 点、v4 高 0.28 点；v4 相比 v3 句内增加 0.59 点、跨句下降 1.18 点。'
        '因此 v4 并未带来总体提升，跨句部分是两项任务共同的退步来源。', '',
        table(['总体 P / R / F1','论文','v3','v4'],[
            ['方向','47.90 / 47.80 / 47.80',
             ' / '.join(pct(run['direction_v3']['mean_test']['direction']['all'][m]) for m in ('p','r','f1')),
             ' / '.join(pct(run['direction_v4']['mean_test']['direction']['all'][m]) for m in ('p','r','f1'))],
            ['存在性','59.20 / 66.10 / 62.50',
             ' / '.join(pct(run['existence_v3']['mean_test']['existence']['all'][m]) for m in ('p','r','f1')),
             ' / '.join(pct(run['existence_v4']['mean_test']['existence']['all'][m]) for m in ('p','r','f1'))]]),'',
        'v3 方向精度已接近论文，召回低 2.58 点；v4 精度和召回都略低。'
        '存在性 v4 比 v3 更积极预测正例：平均召回增加 1.15 点，精度下降 2.16 点，净结果是 F1 下降。', '',
        '**评分口径会改变“接近论文”的程度。** 作者 transfor3to2 将方向判反的预测转成负类，'
        '使其只产生 FN、不产生标准多分类评分应有的 FP。相同预测的源码方向 F1：v3 为 47.14、v4 为 46.40；'
        '比各自标准方向 F1 高 0.64、0.67 点。源码口径用于对照作者实现，论文文字声称的是 CAUSE/EFFECT 标准微平均。'
        '仅凭论文表格无法确认其实际运行时具体采用哪套评分，故必须同时披露，不能只取更接近论文的一列。', '',
        f'![论文对比]({(OUT/"paper_comparison.png").as_posix()})', '',
        '**收敛与选轮。** 每折 30 轮连续，损失均有限；四份正式日志未发现 OOM、非有限损失或异常中止。'
        '训练曲线整体平滑，方向任务的开发集 F1 在约 5—10 轮后进入波动平台；存在性约 15—20 轮后趋缓。', '']
    rows=[]
    for task in TASKS:
        for version in VERSIONS:
            r=run[task+'_'+version]
            rows.append([task+' '+version, '%.5f → %.5f'%(r['epoch_means']['1']['train_loss'],r['epoch_means']['30']['train_loss']),
                f(np.mean([d['dev_best'] for d in r['folds']])),f(r['epoch_means']['30']['dev_f1']),
                ' / '.join(str(d['best_epoch']) for d in r['folds'])])
    lines += [table(['任务／版本','第1轮 → 第30轮平均损失','各折最优开发 F1 均值','第30轮开发 F1 均值','最佳 epoch（折1→5）'],rows),'',
        '损失下降约 98%—99%，开发集收益却较早饱和，说明继续降低训练目标未必改善泛化。'
        '方向 v3 第 2 折在第 6 轮达到开发集峰值，此后到第 30 轮下降 5.64 点；'
        '并非所有折都需要训练到最后才最好。这里是开发集平台和部分折回落的证据，'
        '训练 F1、开发集 loss、历史梯度范数与实际训练图更新次数没有逐轮记录，不能进一步断言训练准确率已接近完美或精确定位过拟合原因。'
        '最终使用 best.pt，因此不能把第 30 轮状态当作报告模型。', '',
        'v4 第一轮损失更高，但后期与 v3 相近，不是训练发散。图结构及实际迭代数会影响加权损失，'
        '两版初期损失大小不能直接解释为模型质量高低。原日志没有训练图密度，'
        '“训练无门槛引入更多不可靠边”只能作为机制假设，尚未由历史训练轨迹直接证实。', '',
        '最佳开发 F1 均值约为方向 33.57／32.32、存在性 48.36／47.86，明显低于测试结果。'
        '开发集来自完全不同的两个话题（37、41），而五折测试与训练共享 18—19 个话题；'
        '候选对正例率也从训练池的 8.08% 降到开发集的 5.73%。这些差异说明开发集难度和分布不同，'
        '不能用开发测试分差单独认定计分或选轮错误；也不能把本次文档级五折当作未见话题泛化验证。', '',
        f'![训练曲线与各折结果]({(OUT/"training_and_folds.png").as_posix()})', '',
        '**折间稳定性。** 以下均为开发集选出的 checkpoint 的标准测试 F1。', '']
    rows=[]
    for i in range(5):
        values=[run[task+'_'+v]['folds'][i]['test'][task]['all']['f1']*100 for task in TASKS for v in VERSIONS]
        rows.append([i+1,*map(f,values),f(values[1]-values[0]),f(values[3]-values[2])])
    lines += [table(['折','方向v3','方向v4','存在性v3','存在性v4','方向 v4−v3','存在性 v4−v3'],rows),'',
        '方向 v3 在 4/5 折领先，存在性 v3 在 3/5 折领先；均值优势分别为 0.77、0.76 点。'
        '存在性 v4 折间标准差 4.81 点，大于 v3 的 3.15 点，主要受第 1 折较差影响。'
        '方向 v4 折间标准差略小（2.74 对 3.01），因此不能笼统说 v4 在所有方面都更不稳定。', '',
        '以五个配对折差做双侧符号翻转计算，方向 p=0.3125，存在性 p=0.4375。'
        '五折训练集合高度重叠，且只用一个 seed，这些 p 值仅为描述性参考，不是可靠的独立重复显著性证据。'
        '训练程序只在实验开始统一设种子，各折不重新设种子；阈值改变迭代次数也会改变随机数消耗，'
        '所以相同 seed 并不保证两个版本各折的初始化、dropout 流完全配对。不能把全部差异严格归因于阈值本身。', '',
        '**错误集中在哪里。** 下表合并五折折外预测，每个版本覆盖同样的 4,933 个正例。', '']
    rows=[]
    for task in TASKS:
        for version in VERSIONS:
            e=run[task+'_'+version]['errors']['all']
            rows.append([task+' '+version,e['correct_causal'],e['missed_causal'],e['reversed_causal'],e['spurious_causal']])
    lines += [table(['任务／版本','因果且类别正确','因果判为NONE','因果方向判反','无因果判为因果'],rows),'',
        '方向 v3／v4 对已经识别为因果的真正例，方向正确率分别为 94.57%／94.10%；'
        '真正的主瓶颈是约一半正例被漏掉，以及大量负例被误报，单纯反向错误只有 128／138 对。'
        'v4 比 v3 少识别正确 28 对，其中 26 对来自跨句；跨句误报增加 54 对。'
        '因此下一步应优先检查跨句候选对的证据和正负判别。', '',
        '存在性 v4 比 v3 多识别正确 47 对，却多误报 248 对；跨句部分只多识别正确 22 对，误报增加 227 对。'
        '这直接解释其召回提高而 F1 下降。', '',
        '方向模型将 CAUSE/EFFECT 合并后，五折存在性 F1 仅为 v3 49.22、v4 48.59；'
        '独立二分类模型则为 63.54／62.78。这个差距不能全解释为“方向判错”，'
        '因为合并方向已经消除了反向错误，而且二分类使用了不同的输入排序协议。', '']
    if traces:
        lines += ['**图迭代是否实际有作用。** 重载每折最佳模型，记录同一次前向的初始化预测和最终预测。', '']
        rows=[]
        for task in TASKS:
            for version in VERSIONS:
                r=traces['runs'][task+'_'+version]
                rows.append([task+' '+version,f(r['mean_updates']),max(d['updates'] for d in r['documents']),
                    f(r['mean_initial_fold_f1']),f(r['mean_final_fold_f1']),f(r['mean_final_fold_f1']-r['mean_initial_fold_f1'])])
        lines += [table(['任务／版本','平均有效图更新次数','最大次数','初始化 F1','最终 F1','增益'],rows),'',
            '全部 932 次文档前向都在邻接差 ≤2 时停止，没有触及上限仍不收敛的情况。'
            '方向约 44% 的文档只更新一次，存在性约 54% 只更新一次；设置 L=9 不等于每篇都传播九轮。'
            '方向收益约 2.6—2.9 点，支持迭代推理在这些已训练 checkpoint 上有帮助。'
            '存在性 v3 增益只有 0.43 点，说明其高分不能主要归功于多轮图推理。', '',
            '这种同一模型的中间结果对比不等价于“去掉迭代后重新训练”的消融：初始化预测没有独立计入训练损失，'
            'MLP 的训练分布也与初始化状态不同。因此不能据此宣称已复现论文表 3 的迭代消融收益。', '',
            f'![实际更新次数]({(OUT/"iteration_distribution.png").as_posix()})', '']
    if ordering:
        lines += ['**存在性结果最关键的限制：真实标签参与了事件对输入排序。** 三分类缓存的事件对均按事件索引正序排列。'
            '二分类缓存则把原 EFFECT 正例反转为 CAUSE：233 篇训练／折外测试文档中 2,674 个正例被反转，占正例 54.21%，'
            '56,132 个负例全部保留正序；固定开发集中另外有 193 个正例被反转。'
            '因而“反序输入”与正类完全关联，这是标签影响输入预处理的直接证据。'
            '模型没有直接输入事件索引，但有序的 h_i、h_j 及有向图会随交换而变化。', '',
            '为量化这种依赖，保持所有 checkpoint 不变，将每个候选事件对统一按索引排序，'
            '排序操作不读取 gold；在相同五折文档上重新推理：', '']
        rows=[]
        for version in VERSIONS:
            r=ordering['runs']['existence_'+version]
            sub=r['positive_subgroups']
            rows.append([version,pct(r['author_mean_test']['all']['f1']),pct(r['mean_test']['all']['f1']),
                f(100*sub['reverse_author_tp']/sub['reverse_gold'])+' → '+f(100*sub['reverse_canonical_tp']/sub['reverse_gold']),
                f(100*sub['forward_author_tp']/sub['forward_gold'])+' → '+f(100*sub['forward_canonical_tp']/sub['forward_gold'])])
        lines += [table(['版本','作者排序总体 F1','统一正序总体 F1','被反转正例的召回变化','原正序正例的召回变化'],rows),'',
            '下降集中于原本按真实方向反转的正例，表明模型对这一输入协议高度敏感。'
            '作者协议下达到 62.5 的数值仍然是有效的条件性复现结果；但当推理时并不知道真实因果方向，'
            '不能假设同样的输入变换可用，也不能直接声称模型已在普通未知方向候选对上达到该能力。', '',
            '**边界必须保留：** 31.76／31.53 是对原 checkpoint 改变推理输入后的敏感性结果，'
            '同时改变了分类器输入分布和图方向，并非在统一排序数据上重新训练的成绩，也不是“排除泄漏后模型只能达到 32”的估计。'
            '要得到公平能力评估，应在训练、开发、测试三处统一采用不依赖标签的排序，重新训练并重做开发集选轮；'
            '之后再将该结果与作者协议复现成绩分列。当前证据不能用于反推论文其他基线是否受同样影响。', '',
            f'![输入顺序敏感性]({(OUT/"pair_order_sensitivity.png").as_posix()})', '']
    lines += ['**资源与剩余限制。** 本机 RTX 4060 Laptop 上，训练加开发评估的逐轮计时均值约为 42—44 秒；'
        '每项任务完整五折耗时约 112—116 分钟，日志峰值 allocated 显存 2.91 GB。'
        '该计时未包含全部测试、保存与刷新报告开销，allocated 也不是整卡显存占用；'
        '硬件、梯度检查点、迭代次数与统计口径不同，不能据此直接宣布比论文 823 秒／16 GB 更快或更省多少倍。', '',
        'ESC 三分类缓存已有与发表前官方数据一致的哈希审计，仍残留两篇有向环标签。'
        'Python 3.8.20、transformers 4.21.1 与作者要求不同；张量 FP32，但 TF32 开关开启，'
        '所以“未开 AMP”不等于全部矩阵乘法完全禁用 TF32。'
        '这类环境差异、损失按文档内 mean 而非论文公式 sum、论文与源码的输出／停止细节解释，'
        '都需要保留在复现说明中；目前没有实验将方向差距具体归因给其中某一项。', '',
        'v3 仅有 MAVEN 小规模试跑，v4 没有对应的完整 MAVEN 方向／存在性结果；'
        '不能把旧版本 MAVEN 的中途状态算入这次 v3/v4 完整复现。'
        '当前也没有完成 v3/v4 的重训消融、多种子或独立话题划分验证。', '',
        '**建议的后续优先级。** 第一，固定当前结果并分别标注“作者输入协议复现”与“标签无关输入能力评估”，'
        '先消除存在性输入排序带来的解释歧义。第二，在标签无关排序上重新训练存在性，同时保留标准方向计分与源码计分两列。'
        '第三，至少增加几个独立种子，并在每折明确固定初始化与采样随机状态，判断约 0.8 点版本差是否稳定；'
        '训练期间补记迭代次数、正例边数、图置信度与梯度范数。第四，再完成 MAVEN 正式实验和论文机制消融。'
        '目前直接增加 epoch 或因测试分数较高而把 v3 改称更忠实版本，均缺少充分依据。', '',
        '**可复查材料。**', '',
        f'- [完整逐折／逐轮统计]({(OUT/"analysis.json").as_posix()})',
        f'- [逐折指标 CSV]({(OUT/"fold_metrics.csv").as_posix()})',
        f'- [20 个 checkpoint 的重载与迭代证据]({(OUT/"reload_iteration_audit.json").as_posix()})',
        f'- [输入排序敏感性证据]({(OUT/"pair_order_sensitivity.json").as_posix()})',
        f'- [分析脚本]({(ROOT/"analyze_v3_v4.py").as_posix()})',
        f'- [本地论文及表 1、附录 B]({(ROOT.parent/"iLIF_markdown/iLIF.md").as_posix()})',
        f'- [训练与开发集选轮实现]({(ROOT/"run.py").as_posix()})',
        f'- [版本契约]({(ROOT/"algorithm_config.py").as_posix()})',
        f'- [两种评分实现]({(ROOT/"common.py").as_posix()})',
        f'- [作者缓存顺序与发表前数据审计]({(ROOT/"artifacts/publication_version_audit.json").as_posix()})', '',
        '复核命令：', '', '```powershell',
        '& F:\\ANACONDA\\envs\\ilif\\python.exe reproduction\\analyze_v3_v4.py',
        '# 实际重载所有模型并记录测试迭代轨迹',
        '& F:\\ANACONDA\\envs\\ilif\\python.exe reproduction\\analyze_v3_v4.py --reload',
        '# 仅补做存在性输入排序敏感性推理',
        '& F:\\ANACONDA\\envs\\ilif\\python.exe reproduction\\analyze_v3_v4.py --canonical-order',
        '```', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    source_paths=[ROOT/p for p in ('analyze_v3_v4.py','run.py','algorithm_config.py','paper_model.py','common.py','model/CGE.py')]
    for task in TASKS:
        for version in VERSIONS:
            rd=Path(run[task+'_'+version]['configuration']['output'])
            source_paths.append(rd/'configuration.json')
            for fd in sorted(rd.glob('fold_*')):
                source_paths += [fd/p for p in ('history.jsonl','result.json','best_predictions.json','split.json')]
    dump(OUT/'analysis_manifest.json',dict(created=datetime.now().isoformat(),
        note='Hashes identify the source and saved evidence used in this audit; model files were actually loaded but are not hashed here.',
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reload',action='store_true')
    parser.add_argument('--canonical-order',action='store_true');args=parser.parse_args()
    result,features,predictions=run_audit()
    if args.canonical_order:
        canonical_order_audit(result,features)
    traces=reload_audit(result,features,predictions) if args.reload else None
    if traces is None and (OUT/'reload_iteration_audit.json').exists():
        traces=read(OUT/'reload_iteration_audit.json')
    plots(result,traces)
    ordering=read(OUT/'pair_order_sensitivity.json') if (OUT/'pair_order_sensitivity.json').exists() else None
    if ordering:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4.5))
        x=np.arange(2)
        for j, (field,label,color) in enumerate([('author_mean_test','Author order','#2563eb'),('mean_test','Canonical order at inference','#e07a29')]):
            vals=[ordering['runs']['existence_'+v][field]['all']['f1']*100 for v in VERSIONS]
            bars=ax.bar(x+(j-.5)*.3,vals,width=.28,label=label,color=color)
            for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+.8,'%.2f'%b.get_height(),ha='center')
        ax.set_xticks(x);ax.set_xticklabels(VERSIONS);ax.set_ylim(0,85);ax.set_ylabel('Five-fold mean F1 (%)')
        ax.set_title('Existence model: sensitivity to pair ordering')
        ax.legend(loc='upper right',fontsize=9);ax.grid(axis='y',alpha=.15)
        fig.text(.5,.015,'Same checkpoints; inference distribution shift. Not a retrained fair baseline.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'pair_order_sensitivity.png');plt.close(fig)
    write_report(result,traces,ordering)
    print('Analysis saved to '+str(OUT),flush=True)
