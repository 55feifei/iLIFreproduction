"""One versioned model contract for training, inference and verification."""
from types import SimpleNamespace
from fusion_config import fusion_from_config

PAPER = 'paper_algorithm_v4'
# Retired, reload only. v3 applied ω during training and evaluation; v2 also
# thresholded the final output at 0.6; v1 had no self-loops, so empty
# neighborhoods became zero.
PAPER_V3 = 'paper_algorithm_v3'
PAPER_V2 = 'paper_algorithm_v2'
PAPER_V1 = 'paper_algorithm_v1'
LEGACY = 'legacy_source'
PAPER_PROFILES = (PAPER, PAPER_V3, PAPER_V2, PAPER_V1)
PROFILES = (PAPER, PAPER_V3, PAPER_V2, PAPER_V1, LEGACY)
TRAINABLE_PROFILES = (PAPER, LEGACY)
PAPER_SUFFIX = '_v4'


def profile(config):
    value = config.get('algorithm_profile', LEGACY)
    if value not in PROFILES:
        raise ValueError('Unknown algorithm profile: ' + str(value))
    return value


def paper_contract(dataset, version=PAPER):
    if version not in PAPER_PROFILES:
        raise ValueError('Not a paper algorithm profile: ' + str(version))
    # Figure 2 draws self-loop weights w_11, w_44 on the intra path only.
    self_loops = 'intra' if version in (PAPER, PAPER_V3, PAPER_V2) else False
    # The 0.6 threshold gates evaluation edges. Table 1 recall (47.8) is not
    # reachable when the reported output is also gated, so output is argmax.
    protocol = 'final_argmax' if version in (PAPER, PAPER_V3) else 'thresholded_final_ecg'
    contract = dict(algorithm_profile=version, graph_orientation='incoming',
        fusion_policy='paper', beta_intra=.7, beta_inter=.3,
        relation_threshold=.6, structural_threshold=2,
        max_graph_updates=9 if dataset == 'esc' else 19,
        num_heads=4, graph_linear_bias=False, mlp_dropout=.4, mlp_activation='leaky_relu',
        leaky_relu_slope=.01, focal_gamma=2., class_weight=.75,
        loss_reduction='mean', initialization='contextual_h',
        initial_prediction_loss=False, self_loops=self_loops,
        isolated_node_rule='intra_self_loop' if self_loops else 'zero_empty_neighborhood',
        prediction_protocol=protocol,
        selection_metric='standard_task_f1', sentence_count='document_sentences',
        plm_lr=2e-5, module_lr=1e-4, warmup_ratio=.1, batch_size=1)
    # Author training uses --rate 0: argmax edges, no ω. v3 kept ω in both modes.
    if version == PAPER:
        contract['train_relation_threshold'] = 0.
    return contract


def validate_contract(config):
    if config.get('ablation', 'none') not in ('none', 'no_direction', 'no_heterogeneity', 'no_iteration'):
        raise ValueError('Unknown ablation: ' + str(config['ablation']))
    if profile(config) in PAPER_PROFILES:
        for key, expected in paper_contract(config['dataset'], profile(config)).items():
            if key not in config or config[key] != expected:
                raise ValueError('Paper algorithm setting mismatch: ' + key)
        if config.get('fusion_version') != 'typed_adjacency_v1':
            raise ValueError('Paper algorithm requires typed fusion')


def validate_resume(previous, requested):
    if profile(previous) != profile(requested):
        raise ValueError('Algorithm changed; use a new experiment directory')
    if previous.get('ablation', 'none') != requested.get('ablation', 'none'):
        raise ValueError('Ablation changed; use independent experiment checkpoints')
    validate_contract(requested)
    if profile(requested) in PAPER_PROFILES:
        validate_contract(previous)
        for key in paper_contract(requested['dataset'], profile(requested)):
            if previous[key] != requested[key]:
                raise ValueError('Resume algorithm setting changed: ' + key)


def selection_metric(config):
    if config['task'] == 'existence':
        return 'existence'
    return 'direction' if profile(config) in PAPER_PROFILES else 'source_direction'


def model_args(config, device):
    validate_contract(config)
    fusion = fusion_from_config(config)
    args = SimpleNamespace(device=device, model_name=config['model_path'], n_last=768,
        mlp_size=2304, mlp_drop=.4, no_of_classes=2 if config['task'] == 'existence' else 3,
        rate=0., w=.6, max_iteration=9 if config['dataset'] == 'esc' else 19,
        threshold=2, min_iteration=2, in_channels={'event': 768},
        metadata=(['event'], [('event', 'intra', 'event'), ('event', 'inter', 'event')]),
        num_heads=4, gamma=2., loss_type='focal', class_weight=.75,
        checkpoint_mlp=config.get('checkpoint_mlp', config['dataset'] == 'maven'),
        resize_vocab_size=config.get('vocab_size', 30522) if config['dataset'] == 'maven' else None,
        ablation=config.get('ablation', 'none'), algorithm_profile=profile(config), **fusion)
    if args.algorithm_profile in PAPER_PROFILES:
        args.num_heads = config['num_heads']
        args.mlp_drop = config['mlp_dropout']
        args.gamma = config['focal_gamma']
        args.class_weight = config['class_weight']
        args.w = config['relation_threshold']
        # Missing on v1–v3 configs: those versions threshold training graphs at ω.
        args.rate = config['train_relation_threshold'] if 'train_relation_threshold' in config else config['relation_threshold']
        args.max_iteration = config['max_graph_updates']
        args.threshold = config['structural_threshold']
        args.min_iteration = 0
        args.leaky_relu_slope = config['leaky_relu_slope']
        args.self_loops = config['self_loops']
        args.prediction_protocol = config['prediction_protocol']
    return args


def build_model(config, device):
    from adapted_model import ReproductionModel
    from paper_model import PaperModel
    args = model_args(config, device)
    model = (PaperModel if profile(config) in PAPER_PROFILES else ReproductionModel)(args).to(device)
    model.CGE.incoming_orientation = config.get('graph_orientation', 'source') == 'incoming'
    return model
