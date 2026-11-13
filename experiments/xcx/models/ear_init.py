"""Deterministic mean-orthogonal EAR gate initialization; ordinary init stays native."""

import math

import torch

from .moe import expert_fc1_weights


def mean_orth_gate(num_experts, offset, count, ffn, hidden, device, std, seed, layer_number):
    if num_experts > hidden:
        raise ValueError("mean_orth requires num_experts <= hidden_size")
    layer_seed = int(seed) + (0 if layer_number is None else int(layer_number)) * 1_000_003
    generator = torch.Generator(device="cpu").manual_seed(layer_seed)
    basis = torch.randn(hidden, num_experts, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(basis, mode="reduced")
    rows = q.T.contiguous()[offset:offset + count].to(device)
    rows = rows * (std / math.sqrt(ffn) * math.sqrt(hidden))
    noise_generator = torch.Generator(device=device).manual_seed(layer_seed + 1 + offset * 9_176)
    noise = torch.randn((count, ffn, hidden), device=device, dtype=torch.float32,
                         generator=noise_generator) * std
    noise = noise - noise.mean(dim=1, keepdim=True)
    return noise + rows[:, None, :]


@torch.no_grad()
def init_ear_gate(layer):
    config, router = layer.config, layer.router
    if router.ear.w_gate_init != "mean_orth" or not config.perform_initialization:
        return
    sources = expert_fc1_weights(layer.experts, layer.num_local_experts,
                                config.moe_ffn_hidden_size, config.hidden_size)
    gate = mean_orth_gate(config.num_moe_experts,
                         router.expert_group.rank() * layer.num_local_experts,
                         layer.num_local_experts, config.moe_ffn_hidden_size,
                         config.hidden_size, sources[0].device, config.init_method_std,
                         router.ear.w_gate_init_seed, layer.layer_number)
    for source, value in zip(sources, gate):
        source[:config.moe_ffn_hidden_size].copy_(value.to(source))
    # YuLan mean_orth intentionally overrides --moe-router-probe-init.
    router.probe_vector.fill_(1. / config.moe_ffn_hidden_size)
