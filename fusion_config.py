"""Versioned fusion settings shared by training, inference and verification."""
import math


def fusion_settings(policy='source', beta_intra=None, beta_inter=None):
    if policy not in ('source', 'paper'):
        raise ValueError('Unknown fusion policy: ' + str(policy))
    if beta_intra is None and beta_inter is None:
        beta_intra, beta_inter = (.7, .3) if policy == 'paper' else (.3, .7)
    elif beta_intra is None:
        beta_intra = 1. - beta_inter
    elif beta_inter is None:
        beta_inter = 1. - beta_intra
    if not all(math.isfinite(x) and 0. <= x <= 1. for x in (beta_intra, beta_inter)):
        raise ValueError('Fusion weights must be finite and between zero and one')
    if not math.isclose(beta_intra + beta_inter, 1., abs_tol=1e-12):
        raise ValueError('Fusion weights must sum to one')
    return dict(fusion_policy=policy, beta_intra=beta_intra, beta_inter=beta_inter,
                fusion_version='typed_adjacency_v1' if policy == 'paper' else 'source_position_v1')


def fusion_from_config(config):
    settings = fusion_settings(config.get('fusion_policy', 'source'),
                               config.get('beta_intra'), config.get('beta_inter'))
    if config.get('fusion_version', settings['fusion_version']) != settings['fusion_version']:
        raise ValueError('Unsupported fusion version')
    return settings


def validate_fusion_resume(previous, requested):
    if fusion_from_config(previous) != fusion_from_config(requested):
        raise ValueError('Resume fusion settings changed; use a new experiment directory')
