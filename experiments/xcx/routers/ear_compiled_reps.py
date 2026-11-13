"""Opt-in gate packing and original-size BMMs; probe casts stay outside compilation."""

import torch


def _pack_gate_weights(weights, ffn):
    # Callers supply explicit [2 * FFN, hidden] FC1 views, with no orientation heuristic.
    return torch.stack([weight[:ffn] for weight in weights])


def _local_rows(probe, weights, ffn, per_expert):
    gate = _pack_gate_weights(weights, ffn)
    p = (probe.unsqueeze(1) if per_expert else
         probe.view(1, 1, -1).expand(len(weights), -1, -1))
    return torch.bmm(p, gate).squeeze(1)


def _stack_rows(probes, sources, ffn, per_expert):
    # Keep each layer's contraction shape, rather than merging into a larger BMM.
    return torch.cat([_local_rows(p, w, ffn, separate)
                      for p, w, separate in zip(probes, sources, per_expert)])


pack_gate_weights = torch.compile(_pack_gate_weights, fullgraph=True)
compiled_local_rows = torch.compile(_local_rows, fullgraph=True)
compiled_stack_rows = torch.compile(_stack_rows, fullgraph=True)
