"""Verify largest real documents fit without truncating; no benchmark scores."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import gc
import argparse
import time
import torch
from common import ROOT, ESC_DATA, batch, load_features, save_json
from legacy_utils import setup_seed
from fusion_config import fusion_settings
from algorithm_config import PAPER, TRAINABLE_PROFILES, PAPER_SUFFIX, paper_contract, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['all', 'esc', 'maven'], default='all')
    parser.add_argument('--algorithm-profile', choices=TRAINABLE_PROFILES, default=PAPER)
    parser.add_argument('--graph-orientation', choices=['source', 'incoming'], default='incoming')
    parser.add_argument('--model-path', default=str(ROOT / 'models/bert-base-uncased'))
    parser.add_argument('--no-checkpoint-bert', action='store_true')
    parser.add_argument('--fusion-policy', choices=['source', 'paper'], default='paper')
    parser.add_argument('--beta-intra', type=float)
    parser.add_argument('--beta-inter', type=float)
    parser.add_argument('--exercise-cap', action='store_true',
                        help='Diagnostic only: force a nonempty graph through every update')
    cli = parser.parse_args()
    fusion = fusion_settings(cli.fusion_policy, cli.beta_intra, cli.beta_inter)
    if cli.algorithm_profile == PAPER and (cli.graph_orientation != 'incoming' or fusion != fusion_settings('paper')):
        parser.error(PAPER + ' requires incoming propagation and paper 0.7/0.3 fusion')
    torch.set_num_threads(4)
    device = torch.device('cuda')
    results = []
    for dataset in (['esc', 'maven'] if cli.dataset == 'all' else [cli.dataset]):
        setup_seed(209)
        config = dict(dataset=dataset, task='direction', model_path=cli.model_path,
            vocab_size=30524 if dataset == 'maven' else 30522,
            checkpoint_mlp=dataset == 'maven', graph_orientation=cli.graph_orientation,
            algorithm_profile=cli.algorithm_profile, **fusion)
        if cli.algorithm_profile == PAPER:
            config.update(paper_contract(dataset))
        model = build_model(config, device)
        if cli.exercise_cap:
            if cli.algorithm_profile != PAPER:
                parser.error('--exercise-cap is a paper-algorithm diagnostic')
            # Isolated stress fixture, not an experiment configuration: retain
            # all update activations and exercise nonempty attention graphs.
            model.threshold = -1
            with torch.no_grad():
                model.mlp[-1].bias[1] = 5.
        if not cli.no_checkpoint_bert:
            model.pretrained_model.gradient_checkpointing_enable()
        model.train()
        # Allocate persistent Adam moments too, not just a forward pass.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        if dataset == 'esc':
            pool = sum([load_features(ESC_DATA / ('EventStoryLine_bert_intra_and_inter_%s_features.pkl' % s))
                        for s in ['train', 'dev']], [])
        else:
            pool = sum([load_features(ROOT / ('data/maven/%s.pkl' % s)) for s in ['train', 'dev', 'test']], [])
        examples = {str(f.doc_id): f for f in [max(pool, key=lambda f: f.enc_input_ids.numel()),
                                             max(pool, key=lambda f: len(f.target))]}
        for doc_id, feature in examples.items():
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            model.zero_grad()
            loss, logits = model(**batch(feature, device))
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            record = dict(dataset=dataset, doc_id=doc_id, input_shape=list(feature.enc_input_ids.shape),
                          vocab_size=model.pretrained_model.config.vocab_size,
                          checkpoint_bert=not cli.no_checkpoint_bert, checkpoint_mlp=config['checkpoint_mlp'],
                          algorithm_profile=cli.algorithm_profile,
                          diagnostic_force_cap=cli.exercise_cap,
                          graph_updates=getattr(model, 'last_graph_updates', None),
                          graph_orientation=cli.graph_orientation,
                          pairs=len(feature.target), seconds=time.monotonic() - start,
                          peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                          peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                          finite_loss=float(loss.detach()))
            results.append(record)
            record.update(fusion)
            print(record, flush=True)
            del loss, logits
            model.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    filename = 'stress.json' if cli.dataset == 'all' else 'stress_%s_vocab%d_%s.json' % (
        cli.dataset, 30524 if cli.dataset == 'maven' else 30522, cli.graph_orientation)
    if cli.fusion_policy == 'paper':
        filename = filename.replace('.json', '_paper.json')
    if cli.algorithm_profile == PAPER:
        filename = filename.replace('.json', PAPER_SUFFIX + '.json')
    if cli.exercise_cap:
        filename = filename.replace('.json', '_cap.json')
    save_json(ROOT / ('artifacts/' + filename), results)


if __name__ == '__main__':
    main()
