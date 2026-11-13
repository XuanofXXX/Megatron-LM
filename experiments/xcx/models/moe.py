"""Expose expert gate views to EAR without registering experts under the Router."""

import torch

from megatron.core.transformer.moe.moe_layer import MoELayer


def expert_fc1_parameters(experts):
    """Return optimizer-owned leaves, rather than the packed/view tensors used by EAR."""
    if hasattr(experts, "local_experts"):
        return tuple(expert.linear_fc1.weight for expert in experts.local_experts)
    if hasattr(experts, "linear_fc1"):
        return tuple(p for name, p in experts.linear_fc1.named_parameters(recurse=False)
                     if name.startswith("weight"))
    return (experts.weight1,)


def expert_fc1_weights(experts, num_local_experts, ffn_hidden_size, hidden_size):
    """Return individual FC1 views for packing/compilation without changing stored params."""
    n, f, h = num_local_experts, ffn_hidden_size, hidden_size
    if hasattr(experts, "local_experts"):
        weights = tuple(expert.linear_fc1.weight for expert in experts.local_experts)
    elif hasattr(experts, "linear_fc1"):
        fc1 = experts.linear_fc1
        weight = getattr(fc1, "weight", None)
        if weight is None:
            weights = tuple(getattr(fc1, f"weight{i}") for i in range(n))
        elif weight.shape in ((n, 2 * f, h), (n * 2 * f, h)):
            weights = weight.reshape(n, 2 * f, h).unbind(0)
        else:
            raise ValueError(f"Unsupported grouped FC1 weight shape: {tuple(weight.shape)}")
    elif hasattr(experts, "weight1") and experts.weight1.shape == (h, n * 2 * f):
        weights = experts.weight1.reshape(h, n, 2 * f).permute(1, 2, 0).unbind(0)
    else:
        raise TypeError(f"EAR cannot read FC1 weights from {type(experts).__name__}")
    if len(weights) != n or any(weight.shape != (2 * f, h) for weight in weights):
        raise ValueError("EAR requires unsharded FC1 expert weights [2 * FFN, hidden]")
    return weights


def expert_gate_weights(experts, num_local_experts, ffn_hidden_size, hidden_size):
    """Read differentiable [local experts, FFN, hidden] gate weights, without caching."""
    n, f, h = num_local_experts, ffn_hidden_size, hidden_size
    if hasattr(experts, "local_experts"):
        weights = [expert.linear_fc1.weight for expert in experts.local_experts]
    elif hasattr(experts, "linear_fc1"):
        fc1 = experts.linear_fc1
        weight = getattr(fc1, "weight", None)
        if weight is not None:
            if weight.shape == (n, 2 * f, h):
                return weight[:, :f, :]
            if weight.shape == (n * 2 * f, h):
                return weight.reshape(n, 2 * f, h)[:, :f, :]
            raise ValueError(f"Unsupported grouped FC1 weight shape: {tuple(weight.shape)}")
        weights = [getattr(fc1, f"weight{i}") for i in range(n)]
    elif hasattr(experts, "weight1"):
        # Compatibility with the legacy grouped-GEMM [hidden, experts * 2 * FFN] layout.
        weight = experts.weight1
        if weight.shape != (h, n * 2 * f):
            raise ValueError(f"Unsupported grouped weight1 shape: {tuple(weight.shape)}")
        return weight.reshape(h, n, 2, f).permute(1, 2, 3, 0)[:, 0]
    else:
        raise TypeError(f"EAR cannot read gate weights from {type(experts).__name__}")
    if len(weights) != n or any(weight.shape != (2 * f, h) for weight in weights):
        raise ValueError("EAR requires unsharded FC1 expert weights with shape [2 * FFN, hidden]")
    return torch.stack([weight[:f] for weight in weights])


class EARMoELayer(MoELayer):
    """Keep the official MoE forward/dispatcher and bind its expert weight provider."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.router.bind_experts(self.experts, self.num_local_experts)
        self._ear_init_done = False
        self._maybe_init_ear()

    def _maybe_init_ear(self):
        if not self._ear_init_done and self.layer_number is not None:
            from .ear_init import init_ear_gate

            init_ear_gate(self)
            self._ear_init_done = True

    def set_layer_number(self, layer_number):
        super().set_layer_number(layer_number)
        if hasattr(self, "_ear_init_done"):
            self._maybe_init_ear()
