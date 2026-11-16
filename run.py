"""Resumable source reproduction. All reports distinguish source and standard metrics."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
import platform
import sys
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import KFold
from transformers import AdamW, get_linear_schedule_with_warmup
from common import ROOT, ESC_DATA, batch, load_features, save_json, scores
from legacy_utils import setup_seed
from report import main as refresh_report
from fusion_config import fusion_settings, validate_fusion_resume
from algorithm_config import (PAPER, LEGACY, TRAINABLE_PROFILES, PAPER_SUFFIX, paper_contract,
                              profile, validate_resume, build_model, selection_metric)
from ablation_preflight import sha256


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def evaluate(model, features, device):
    model.eval()
    gold, pred, types, documents = [], [], [], []
    with torch.no_grad():
        for f in features:
            _, logits = model(**batch(f, device))
            argmax = logits.argmax(-1).cpu().tolist()
            protocol = getattr(model, 'prediction_protocol', None)
            p = model.output_labels(logits).cpu().tolist() if protocol == 'thresholded_final_ecg' else argmax
            gold.extend(map(int, f.target))
            pred.extend(p)
            types.extend(map(int, f.rel_type))
            confidence = logits.softmax(-1).max(-1).values.cpu().tolist()
            graph = model.get_graphedge_index(logits, f.event_pairs, [f.rel_type])
            documents.append(dict(doc_id=str(f.doc_id), gold=list(map(int, f.target)),
                pred=p, confidence=confidence, rel_type=list(map(int, f.rel_type)),
                event_pairs=[list(map(int, pair)) for pair in f.event_pairs],
                final_ecg={key[1]: value.t().cpu().tolist() for key, value in graph.items()}))
            if protocol:
                documents[-1].update(argmax_pred=argmax, relation_threshold=model.w,
                    prediction_protocol=protocol)
    return scores(gold, pred, types), documents


def atomic_torch_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def restore_rng(state):
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def rng_fingerprint():
    return dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                cuda=[hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
                      for value in torch.cuda.get_rng_state_all()])


def parameter_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode('utf-8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['esc', 'maven'], default='esc')
    parser.add_argument('--task', choices=['direction', 'existence'], default='direction')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--seed', type=int, default=209)
    parser.add_argument('--limit-train', type=int, default=0)
    parser.add_argument('--limit-eval', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-bert', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--model-path', type=str, default=str(ROOT / 'models/bert-base-uncased'))
    parser.add_argument('--algorithm-profile', choices=TRAINABLE_PROFILES, default=PAPER)
    parser.add_argument('--graph-orientation', choices=['source', 'incoming'])
    parser.add_argument('--fusion-policy', choices=['source', 'paper'])
    parser.add_argument('--beta-intra', type=float)
    parser.add_argument('--beta-inter', type=float)
    parser.add_argument('--ablation', choices=['none', 'no_direction', 'no_heterogeneity', 'no_iteration'], default='none')
    parser.add_argument('--no-refresh-report', action='store_true',
                        help='Keep a separately managed batch from rewriting historical reports')
    cli = parser.parse_args()
    if cli.graph_orientation is None:
        cli.graph_orientation = 'incoming' if cli.algorithm_profile == PAPER else 'source'
    if cli.fusion_policy is None:
        cli.fusion_policy = 'paper' if cli.algorithm_profile == PAPER else 'source'
    fusion = fusion_settings(cli.fusion_policy, cli.beta_intra, cli.beta_inter)
    if cli.algorithm_profile == PAPER:
        if cli.graph_orientation != 'incoming' or fusion != fusion_settings('paper'):
            parser.error(PAPER + ' requires incoming propagation and paper 0.7/0.3 fusion')
    if cli.dataset == 'maven' and cli.task == 'existence':
        parser.error('MAVEN existence is derived from the direction model, as in the paper')
    if cli.dataset == 'maven' and cli.ablation != 'none':
        parser.error('The paper Table 3 ablations use EventStoryLine')
    torch.set_num_threads(cli.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Full reproduction requires CUDA in the ilif environment')
    setup_seed(cli.seed)
    small = cli.limit_train > 0 or cli.limit_eval > 0 or cli.epochs != 30
    output = cli.output or ROOT / 'runs' / ('%s_%s_%s' % (cli.dataset, cli.task, 'pilot' if small else 'full'))
    if cli.output is None and cli.graph_orientation != 'source':
        output = output.with_name(output.name + '_' + cli.graph_orientation)
    if cli.output is None and cli.fusion_policy == 'paper':
        output = output.with_name(output.name + '_paper')
    if cli.output is None and cli.algorithm_profile == PAPER:
        output = output.with_name(output.name + PAPER_SUFFIX)
    if cli.output is None and cli.ablation != 'none':
        output = output.with_name(output.name + '_' + cli.ablation)
    output.mkdir(parents=True, exist_ok=True)
    configuration = vars(cli).copy()
    configuration['output'] = str(output)
    configuration.update(fusion)
    if cli.algorithm_profile == PAPER:
        configuration.update(paper_contract(cli.dataset))
    configuration.update(device=str(device), checkpoint_profile='released_source' if cli.graph_orientation == 'source' else 'source_with_incoming_graph',
                         partial_experiment=small, python=sys.executable)
    if cli.algorithm_profile == PAPER:
        configuration['checkpoint_profile'] = PAPER
    configuration['preprocessing'] = 'author_ESC_cache' if cli.dataset == 'esc' else 'author_labels_with_aligned_relation_types'
    configuration['checkpoint_mlp'] = cli.dataset == 'maven'
    configuration['vocab_size'] = 30524 if cli.dataset == 'maven' else 30522
    source_files = ['run.py', 'common.py', 'project_paths.py', 'algorithm_config.py',
                    'fusion_config.py', 'paper_model.py', 'adapted_model.py',
                    'model/model.py', 'model/CGE.py', 'legacy_utils.py',
                    'ablation_preflight.py']
    configuration['source_sha256'] = {name: sha256(ROOT / name) for name in source_files}
    properties = torch.cuda.get_device_properties(0)
    configuration['runtime'] = dict(platform=platform.platform(), torch=torch.__version__,
        cuda=torch.version.cuda, gpu=properties.name, capability=list(torch.cuda.get_device_capability()),
        multiprocessors=properties.multi_processor_count,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        fold_rng_policy='continuous_across_folds_as_historical_local_v4')
    if cli.dataset == 'esc':
        data_directory = ESC_DATA / 'exist' if cli.task == 'existence' else ESC_DATA
        configuration['esc_data'] = str(ESC_DATA)
        configuration['data_sha256'] = {split: sha256(data_directory /
            ('EventStoryLine_bert_intra_and_inter_%s_features.pkl' % split)) for split in ('train', 'dev')}
        configuration['model_weights_sha256'] = sha256(Path(cli.model_path) / 'pytorch_model.bin')
    configuration['numeric_precision'] = dict(
        parameter_dtype=str(torch.get_default_dtype()), mixed_precision=False,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_enabled=torch.backends.cudnn.enabled)
    configuration['graph_semantics_reference'] = (
        'https://github.com/LchengC/iLIF/blob/f00df860f372eaa7fdb5924334ba0f50d41e3672/model/CGE.py'
        if cli.graph_orientation == 'incoming' else
        'https://github.com/LchengC/iLIF/blob/8769cab82f00eb7fbfe9c4223253f69832105980/model/CGE.py')
    config_path = output / 'configuration.json'
    if not config_path.exists() and any(output.glob('fold_*/*.pt')):
        raise ValueError('Existing checkpoints lack a configuration; use a new experiment directory')
    if config_path.exists() and not cli.resume:
        raise ValueError('Experiment already exists: use --resume or a new --output directory')
    if cli.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        validate_resume(previous, configuration)
        validate_fusion_resume(previous, configuration)
        for field in ('data_sha256', 'model_weights_sha256', 'source_sha256'):
            if field in previous and previous[field] != configuration.get(field):
                raise ValueError('Resume provenance changed: ' + field)
        for field in ['dataset', 'task', 'epochs', 'seed', 'limit_train', 'limit_eval', 'checkpoint_bert', 'model_path', 'graph_orientation', 'vocab_size', 'ablation']:
            defaults = {'graph_orientation': 'source', 'vocab_size': 30522, 'ablation': 'none'}
            old_value = previous.get(field, defaults.get(field))
            if old_value != configuration[field]:
                raise ValueError('Resume configuration changed: ' + field)
    save_json(config_path, configuration)
    if cli.dataset == 'maven' and not small:
        fusion_suffix = '_paper' + PAPER_SUFFIX if cli.algorithm_profile == PAPER else ('_paper' if cli.fusion_policy == 'paper' else '')
        stress_path = ROOT / ('artifacts/stress_maven_vocab30524_%s%s.json' % (cli.graph_orientation, fusion_suffix))
        verified = json.loads(stress_path.read_text(encoding='utf-8')) if stress_path.exists() else []
        if not verified or any(record.get('algorithm_profile', LEGACY) != cli.algorithm_profile or
                               record.get('checkpoint_bert') != cli.checkpoint_bert or
                               fusion_settings(**{key: record[key] for key in
                                   ['beta_intra', 'beta_inter'] if key in record},
                                   policy=record.get('fusion_policy', 'source')) != fusion
                               for record in verified):
            log('Verify largest MAVEN document with the selected vocabulary and checkpoint settings')
            command = [sys.executable, str(ROOT / 'stress.py'), '--dataset', 'maven',
                       '--graph-orientation', cli.graph_orientation, '--model-path', cli.model_path,
                       '--fusion-policy', cli.fusion_policy,
                       '--algorithm-profile', cli.algorithm_profile,
                       '--beta-intra', str(fusion['beta_intra']), '--beta-inter', str(fusion['beta_inter'])]
            if not cli.checkpoint_bert:
                command.append('--no-checkpoint-bert')
            subprocess.run(command, check=True)
    if cli.dataset == 'esc':
        directory = ESC_DATA / 'exist' if cli.task == 'existence' else ESC_DATA
        pool = load_features(directory / 'EventStoryLine_bert_intra_and_inter_train_features.pkl')
        dev = load_features(directory / 'EventStoryLine_bert_intra_and_inter_dev_features.pkl')
        splits = list(KFold(5, shuffle=True, random_state=cli.seed).split(pool))
    else:
        directory = ROOT / 'data/maven'
        pool, dev, test = [load_features(directory / (s + '.pkl')) for s in ['train', 'dev', 'test']]
        splits = [(np.arange(len(pool)), None)]
        cli.folds = [1]
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        if fold not in cli.folds:
            continue
        fold_dir = output / ('fold_%d' % fold)
        fold_dir.mkdir(exist_ok=True)
        if cli.resume and (fold_dir / 'result.json').exists():
            fold_results.append(json.loads((fold_dir / 'result.json').read_text(encoding='utf-8')))
            rng_path = fold_dir / 'rng.pt'
            if not rng_path.exists():
                raise RuntimeError('Completed fold lacks RNG state: ' + str(fold_dir))
            restore_rng(torch.load(rng_path, map_location='cpu'))
            log('Skip completed fold %d' % fold)
            continue
        train = [pool[i] for i in train_idx]
        heldout = [pool[i] for i in test_idx] if test_idx is not None else test
        valid = dev
        if cli.limit_train:
            train = train[:cli.limit_train]
        if cli.limit_eval:
            heldout, valid = heldout[:cli.limit_eval], valid[:cli.limit_eval]
        save_json(fold_dir / 'split.json', dict(train=[str(f.doc_id) for f in train],
                  dev=[str(f.doc_id) for f in valid], test=[str(f.doc_id) for f in heldout]))
        model = build_model(configuration, device)
        if cli.checkpoint_bert:
            model.pretrained_model.gradient_checkpointing_enable()
        if not (fold_dir / 'initialization_fingerprint.json').exists():
            rng_before = rng_fingerprint()
            initialization = dict(parameters_sha256=parameter_fingerprint(model), rng=rng_before,
                                  seed=cli.seed, fold=fold, task=cli.task, ablation=cli.ablation)
            assert rng_before == rng_fingerprint(), 'Fingerprint must not consume random numbers'
            save_json(fold_dir / 'initialization_fingerprint.json', initialization)
        groups = []
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        plm_lr = configuration.get('plm_lr', 2e-5)
        module_lr = configuration.get('module_lr', 1e-4)
        for module, lr in [(model.pretrained_model, plm_lr), (model.mlp, module_lr), (model.CGE, module_lr)]:
            for decay in [True, False]:
                groups.append(dict(params=[p for n, p in module.named_parameters()
                    if any(key in n for key in no_decay) != decay],
                    weight_decay=.01 if decay else 0., lr=lr))
        optimizer = AdamW(groups, lr=module_lr, eps=1e-8)
        total_steps = len(train) * cli.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, total_steps * configuration.get('warmup_ratio', .1), total_steps)
        start_epoch, best_score, best = 1, -1., None
        last_path = fold_dir / 'last.pt'
        if cli.resume and last_path.exists():
            state = torch.load(last_path, map_location='cpu')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            start_epoch, best_score, best = state['epoch'] + 1, state['best_score'], state['best']
            restore_rng(state)
            del state
        for epoch in range(start_epoch, cli.epochs + 1):
            start = time.monotonic()
            model.train()
            model.zero_grad()
            losses = []
            document_order, graph_updates = [], Counter()
            epoch_rng = rng_fingerprint()
            # Match RandomSampler's RNG consumption in the released DataLoader.
            torch.empty((), dtype=torch.int64).random_().item()
            sampler = torch.utils.data.RandomSampler(train)
            for step, index in enumerate(sampler, 1):
                feature = train[index]
                loss, logits = model(**batch(feature, device))
                document_order.append(str(feature.doc_id))
                graph_updates[str(getattr(model, 'last_graph_updates', -1))] += 1
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss at ' + str(feature.doc_id))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                losses.append(float(loss.detach()))
                if step == 1 or step % 25 == 0 or step == len(train):
                    elapsed = time.monotonic() - start
                    log('fold=%d epoch=%d step=%d/%d loss=%.5f seconds=%.1f peak_GB=%.2f' %
                        (fold, epoch, step, len(train), np.mean(losses), elapsed,
                         torch.cuda.max_memory_allocated() / 1e9))
                    save_json(output / 'progress.json', dict(fold=fold, epoch=epoch,
                        step=step, steps=len(train), seconds=elapsed, status='training'))
            # Use only development set for epoch selection; evaluate test on improvement.
            torch.empty((), dtype=torch.int64).random_().item()  # source test loader
            torch.empty((), dtype=torch.int64).random_().item()  # source dev loader
            valid_scores, _ = evaluate(model, valid, device)
            selection = selection_metric(configuration)
            value = valid_scores[selection]['all']['f1']
            entry = dict(epoch=epoch, train_loss=float(np.mean(losses)),
                         seconds=time.monotonic() - start, validation=valid_scores,
                         train_doc_ids=document_order, graph_update_counts=dict(graph_updates),
                         epoch_start_rng=epoch_rng)
            if value >= best_score:
                test_scores, predictions = evaluate(model, heldout, device)
                best_score = value
                best = dict(fold=fold, epoch=epoch, validation=valid_scores,
                            test=test_scores, partial_experiment=small)
                atomic_torch_save(model.state_dict(), fold_dir / 'best.pt')
                save_json(fold_dir / 'best_predictions.json', predictions)
                save_json(fold_dir / 'best.json', best)
            with (fold_dir / 'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry) + '\n')
            atomic_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), epoch=epoch, best_score=best_score, best=best,
                random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all()), last_path)
            log('fold=%d epoch=%d dev_F1=%.4f best_test_F1=%.4f' %
                (fold, epoch, value, best['test'][selection]['all']['f1']))
            if not cli.no_refresh_report:
                refresh_report()
        atomic_torch_save(dict(random=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()), fold_dir / 'rng.pt')
        save_json(fold_dir / 'result.json', best)
        if not cli.no_refresh_report:
            refresh_report()
        fold_results.append(best)
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    summary = dict(folds=fold_results, partial_experiment=small,
                   completed_folds=len(fold_results), expected_folds=5 if cli.dataset == 'esc' else 1)
    if fold_results:
        summary['mean_test'] = {mode: {scope: {metric: float(np.mean([
            f['test'][mode][scope][metric] for f in fold_results]))
            for metric in ['p', 'r', 'f1']} for scope in ['all', 'intra', 'inter']}
            for mode in ['direction', 'existence', 'source_direction']}
    save_json(output / 'summary.json', summary)
    save_json(output / 'progress.json', dict(status='complete', completed_folds=len(fold_results), partial_experiment=small))
    log(json.dumps(summary.get('mean_test', {})))


if __name__ == '__main__':
    main()
