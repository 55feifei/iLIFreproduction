"""Paper equations (4), (6), (7), (8), separate from the archived source path.

Initialization makes G0 from z0=h without an extra learned projection. Update
l=1..min(document sentences,L) uses G(l-1), predicts G(l), and adds loss/l.
G0 is used for graph construction only; equation (8) starts at l=1.
Message passing adds a self-loop to every node on the intra path, as drawn in
Figure 2 (w_11, w_44); the ECG itself and equation (6) exclude these loops.
Evaluation builds every iteration's ECG with ω (confidence >= 0.6). Training
uses args.rate: 0 matches the author --rate default and keeps argmax edges
with no ω. Reported labels follow args.prediction_protocol (argmax for v3/v4).
"""
import torch
from numbers import Integral
from torch import nn
from torch.utils.checkpoint import checkpoint
from adapted_model import ReproductionModel
from model.model import focal_loss


class PaperModel(ReproductionModel):
    def __init__(self, args):
        super().__init__(args)
        self.mlp[1] = nn.LeakyReLU(negative_slope=args.leaky_relu_slope)
        # The graph transform in equation (3) is Wz, without an added bias.
        self.CGE.proj['event'].bias = None
        self.focal_loss = focal_loss(alpha=args.class_weight, gamma=args.gamma,
                                    num_classes=args.no_of_classes, size_average=True)
        self.max_graph_updates = args.max_iteration
        self.max_iteration = args.max_iteration
        self.threshold = args.threshold
        self.self_loops = args.self_loops
        self.prediction_protocol = args.prediction_protocol
        self.last_graph_updates = 0

    def classify_nodes(self, contextual_pairs, nodes, pairs):
        difference = nodes[pairs[:, 0]] - nodes[pairs[:, 1]]
        return self.classify(contextual_pairs, difference)

    def predict(self, contextual_pairs, nodes, pairs):
        if self.training and self.checkpoint_mlp:
            # Save node representations, rather than a large pair-difference
            # tensor for every iteration. Recompute differences during backward.
            return checkpoint(self.classify_nodes, contextual_pairs, nodes, pairs)
        return self.classify_nodes(contextual_pairs, nodes, pairs)

    @torch.no_grad()
    def output_labels(self, prediction):
        confidence, labels = prediction.softmax(-1).max(-1)
        return labels.masked_fill(confidence < self.w, 0)

    @torch.no_grad()
    def iteration_labels(self, prediction):
        # Author get_graphedge_index: training reads --rate, evaluation reads ω.
        confidence, labels = prediction.softmax(-1).max(-1)
        rate = self.rate if self.training else self.w
        if rate:
            labels = labels.masked_fill(confidence < rate, 0)
        return labels

    @torch.no_grad()
    def get_graphedge_index(self, prediction, event_pair, rel_type):
        labels = self.iteration_labels(prediction)
        pairs = torch.as_tensor(event_pair, device=prediction.device, dtype=torch.long).reshape(-1, 2)
        types = torch.as_tensor(rel_type[0], device=prediction.device)
        edges = {}
        for kind, name in [(0, 'intra'), (1, 'inter')]:
            forward = pairs[(types == kind) & (labels == 1)]
            reverse = pairs[(types == kind) & (labels == 2)].flip(-1)
            edge = torch.cat([forward, reverse], dim=0).t().contiguous()
            if edge.numel():
                edges[('event', name, 'event')] = edge
        if self.ablation == 'no_direction':
            edges = {kind: torch.cat([edge, edge.flip(0)], dim=1) for kind, edge in edges.items()}
        elif self.ablation == 'no_heterogeneity' and edges:
            edges = {('event', 'intra', 'event'): torch.cat(list(edges.values()), dim=1)}
        return edges

    @torch.no_grad()
    def message_graph(self, graph, node_count, device):
        if self.self_loops != 'intra':
            return graph
        key = ('event', 'intra', 'event')
        loops = torch.arange(node_count, device=device).repeat(2, 1)
        graph = dict(graph)
        graph[key] = torch.cat([graph[key], loops], dim=1) if key in graph else loops
        return graph

    @staticmethod
    @torch.no_grad()
    def graph_difference(previous, current, node_count):
        def entries(graph):
            if not graph:
                return torch.empty(0, dtype=torch.long)
            return torch.unique(torch.cat([edge[0] * node_count + edge[1] for edge in graph.values()]))
        before, after = entries(previous), entries(current)
        if not before.numel():
            return int(after.numel())
        if not after.numel():
            return int(before.numel())
        # Cardinality of the symmetric difference of directed adjacency entries.
        return 2 * int(torch.unique(torch.cat([before, after])).numel()) - before.numel() - after.numel()

    def forward(self, enc_input_ids, enc_mask_ids, node_event, t1_pos, t2_pos,
                target, rel_type, event_pairs, sentence_count=None):
        if sentence_count is None or sentence_count < 1:
            raise ValueError('Paper iterations require the original document sentence count')
        sent_emb = self.pretrained_model(enc_input_ids, enc_mask_ids)[0]
        contextual = []
        for event in node_event:
            occurrences = [event] if isinstance(event[0], Integral) else event
            contextual.append(torch.stack([self.extract_event(sent_emb[int(sid)], span)
                for sid, span in occurrences]).mean(0))
        h = torch.cat(contextual, dim=0)
        pairs = torch.as_tensor(event_pairs, device=h.device, dtype=torch.long).reshape(-1, 2)
        contextual_pairs = torch.cat([h[pairs[:, 0]], h[pairs[:, 1]]], dim=-1)
        targets = torch.as_tensor(target[0], device=h.device, dtype=torch.long)
        z = h  # G0 uses the contextual representation, as stated in section 3.4.
        logits = self.predict(contextual_pairs, z, pairs)
        graph = self.get_graphedge_index(logits, event_pairs, rel_type)
        if self.record_trace:
            self.iteration_trace = [logits.detach().cpu()]
        limit = min(int(sentence_count), self.max_graph_updates)
        if self.ablation == 'no_iteration':
            limit = 1
        loss = h.new_zeros(())
        for iteration in range(1, limit + 1):
            messages = self.message_graph(graph, h.shape[0], h.device)
            if messages:
                z = self.CGE({'event': z}, messages)['event']
            else:
                # Equation (3): the sum over an empty neighborhood is zero.
                z = torch.zeros_like(z)
            logits = self.predict(contextual_pairs, z, pairs)
            loss = loss + self.focal_loss(logits, targets) / iteration
            updated = self.get_graphedge_index(logits, event_pairs, rel_type)
            difference = self.graph_difference(graph, updated, h.shape[0])
            graph = updated
            self.last_graph_updates = iteration
            if self.record_trace:
                self.iteration_trace.append(logits.detach().cpu())
            if difference <= self.threshold:
                break
        return loss, logits
