"""Execution improvements preserving released model behavior by default.

Reference source stays untouched. Vectorized graph construction preserves
relation insertion order and class ordering; source stopping behavior is kept.
"""
import torch
from model.model import bertCSRModel


class ReproductionModel(bertCSRModel):
    def __init__(self, args):
        super().__init__(args)
        self.ablation = getattr(args, 'ablation', 'none')
        if self.ablation == 'no_iteration':
            # Initial classification -> one graph update -> final classification.
            self.fixed_prediction_rounds = 2

    @torch.no_grad()
    def get_graphedge_index(self, prediction, event_pair, rel_type):
        rate = self.rate if self.training else self.w
        confidence, labels = prediction.softmax(-1).max(-1)
        pairs = torch.as_tensor(event_pair, device=prediction.device, dtype=torch.long)
        types = torch.as_tensor(rel_type[0], device=prediction.device)
        edges = {}
        for kind, name in [(0, 'intra'), (1, 'inter')]:
            eligible = (types == kind) & (confidence > rate)
            forward = pairs[eligible & (labels == 1)]
            reverse = pairs[eligible & (labels == 2)].flip(-1)
            edge = torch.cat([forward, reverse], dim=0).t().contiguous()
            if edge.numel():
                edges[('event', name, 'event')] = edge
        ablation = getattr(self, 'ablation', 'none')
        if ablation == 'no_direction':
            edges = {kind: torch.cat([index, index.flip(0)], dim=1)
                     for kind, index in edges.items()}
        elif ablation == 'no_heterogeneity' and edges:
            edges = {('event', 'intra', 'event'): torch.cat(list(edges.values()), dim=1)}
        return edges

    @torch.no_grad()
    def Contrast_pre(self, prediction, prediction_last):
        rate = self.rate if self.training else self.w
        # Source uses self.rate here even during evaluation; preserve it.
        labels = []
        for logits in (prediction, prediction_last):
            if rate:
                confidence, label = logits.softmax(-1).max(-1)
                label = label.masked_fill(confidence <= self.rate, 0)
            else:
                label = logits.argmax(-1)
            if getattr(self, 'ablation', 'none') == 'no_direction':
                label = (label > 0).long()
            labels.append(label)
        return int((labels[0] != labels[1]).sum())
