"""Small, mergeable summaries for parameter, activation and router diagnostics."""

import math

import torch
import torch.distributed as dist

STAT_NAMES = ("mean", "std", "rms", "l2norm", "l1", "max", "min", "absmax")
ROUTER_METRICS = (
    "router_logits_entropy", "router_logits_entropy_normalized",
    "router_logits_topk_entropy", "router_logits_topk_entropy_normalized",
    "router_probs_topk_entropy", "router_probs_topk_entropy_normalized",
    "router_probs_top1_margin", "router_logits_topk_margin",
)


@torch.no_grad()
def summarize(tensor, *, hidden=False, shard_group=None):
    """Summarize a parameter, or average activation statistics over last-dimension rows.

    Only the last dimension may be sharded here. Activation rows must be aligned across
    that group. Other distributed axes are handled when compact summaries are merged.
    """
    x = tensor.detach().float()
    if x.numel() == 0:
        return None
    rows = x.reshape(-1, x.shape[-1]) if hidden and x.ndim else x.reshape(1, -1)
    variance, local_mean = torch.var_mean(rows, dim=-1, correction=0)
    local_count = torch.full_like(local_mean, rows.shape[-1])
    moments = torch.stack((
        local_count, local_mean * local_count,
        rows.square().sum(-1), rows.abs().sum(-1),
    ))
    extrema = torch.stack((x.max(), -x.min()))
    if shard_group is not None and shard_group.size() > 1:
        dist.all_reduce(moments, group=shard_group)
        dist.all_reduce(extrema, op=dist.ReduceOp.MAX, group=shard_group)
    count, total, squares, absolute = moments
    mean = total / count
    centered = variance * local_count
    if shard_group is not None and shard_group.size() > 1:
        # Parallel variance merge; subtraction of two large raw moments loses small stds.
        centered = centered + local_count * (local_mean - mean).square()
        dist.all_reduce(centered, group=shard_group)
    std = (centered / (count - 1).clamp_min(1)).sqrt()
    return torch.stack((
        mean.new_tensor(mean.numel()), mean.sum(), std.sum(),
        (squares / count).sqrt().sum(), squares.sqrt().sum(),
        (absolute / count).sum(), extrema[0], -extrema[1],
    ))


def merge_summary(left, right):
    """Merge summaries with weighted averages and population extrema."""
    result = left + right
    result[6] = torch.maximum(left[6], right[6])
    result[7] = torch.minimum(left[7], right[7])
    return result


def summary_values(summary):
    count, mean, std, rms, l2, l1, maximum, minimum = summary.tolist()
    return dict(zip(STAT_NAMES, (
        mean / count, std / count, rms / count, l2 / count, l1 / count,
        maximum, minimum, max(abs(maximum), abs(minimum)),
    )))


def statistic_tags(summaries):
    """Keep YuLan's public parameter/hidden-state tag layout."""
    values, totals = {}, {}
    for (kind, name, layer), summary in sorted(summaries.items()):
        for stat, value in summary_values(summary).items():
            if kind == "param" and stat == "l2norm":
                continue
            suffix = f"{name}/layer_{layer:02d}" if kind == "param" else (
                f"_layer_{layer:02d}_{name}"
            )
            values[f"{kind}_{stat}/{suffix}"] = value
        key = kind, name
        totals[key] = merge_summary(totals[key], summary) if key in totals else summary
    for (kind, name), summary in totals.items():
        for stat, value in summary_values(summary).items():
            if kind != "param" or stat != "l2norm":
                values[f"{kind}_{stat}/{name}"] = value
    return values


@torch.no_grad()
def router_statistics(logits, probs, routing_map, names, topk):
    """Return per-token values; the caller retains only their sum and population count."""
    result = {}
    p = probs.detach().float()
    mask = routing_map.bool()

    def add_entropy(prefix, scores, selected=None):
        if not any(name.startswith(prefix + "_entropy") for name in names):
            return
        if selected is not None:
            scores = scores.masked_fill(~selected, 0)
        valid = scores.sum(-1) > 0
        scores = scores[valid]
        scores = scores / scores.sum(-1, keepdim=True).clamp_min(1e-30)
        entropy = -(scores * scores.clamp_min(1e-30).log()).sum(-1)
        result[prefix + "_entropy"] = entropy
        width = scores.shape[-1] if selected is None else selected[valid].sum(-1).float()
        denominator = math.log(width) if isinstance(width, int) and width > 1 else (
            width.clamp_min(2).log() if isinstance(width, torch.Tensor) else 1.0
        )
        result[prefix + "_entropy_normalized"] = entropy / denominator

    if logits is not None:
        z = logits.detach().float().reshape_as(p)
        add_entropy("router_logits", z.softmax(-1))
        if any(name.startswith("router_logits_topk_entropy") for name in names):
            # Use the actual selected experts, including bias correction and token dropping.
            selected = z.masked_fill(~mask, float("-inf"))
            valid = mask.any(-1)
            selected = selected[valid].softmax(-1)
            add_entropy("router_logits_topk", selected, mask[valid])
        if "router_logits_topk_margin" in names:
            result["router_logits_topk_margin"] = (
                z.topk(topk + 1, dim=-1).values[:, topk - 1:topk + 1].diff().neg().flatten()
                if topk < z.shape[-1] else z.new_zeros(z.shape[0])
            )
    add_entropy("router_probs_topk", p, mask)
    if "router_probs_top1_margin" in names:
        p = p.masked_fill(~mask, 0)
        p = p[p.sum(-1) > 0]
        result["router_probs_top1_margin"] = (
            p.topk(2, dim=-1).values.diff().neg().flatten()
            if p.shape[-1] > 1 else p.flatten()
        )
    return {name: value for name, value in result.items() if name in names and value.numel()}


def load_balance_tags(loads):
    """Summarize complete-step hard counts and probability mass, preserving old tags."""
    if not loads:
        return {}
    result = {}
    counts = torch.stack([value[0] for _, value in sorted(loads.items())]).double()
    mass = torch.stack([value[1] for _, value in sorted(loads.items())]).double()
    valid = counts.sum(-1) > 0
    count_values = counts[valid]
    for suffix, population in (("", count_values), ("_mass", mass[mass.sum(-1) > 0])):
        if not population.numel():
            continue
        relative = population / population.mean(-1, keepdim=True) - 1
        hi, lo = relative.max(-1).values, relative.min(-1).values
        for name, value in (
            ("max_vio", hi.mean()), ("min_vio", lo.mean()),
            ("max_max_vio", hi.max()), ("min_min_vio", lo.min()),
        ):
            result[name + suffix] = value.item()
            result["load_balance/" + name + suffix] = value.item()
    if count_values.numel():
        average = count_values.mean(-1, keepdim=True)
        for name, value in (
            ("min_tokens_per_expert", count_values.min()),
            ("mean_min_tokens_per_layer", count_values.min(-1).values.mean()),
            ("max_tokens_per_expert", count_values.max()),
            ("dead_expert_count", (count_values == 0).sum()),
            ("dead_expert_fraction", (count_values == 0).double().mean()),
            ("low_load_expert_fraction_10pct", (count_values < 0.1 * average).double().mean()),
            ("load_cv", (count_values.std(-1, unbiased=False) / average.flatten()).mean()),
        ):
            result["load_balance/" + name] = value.item()
    for layer, (count, _) in loads.items():
        if count.sum() > 0:
            relative = count / count.mean() - 1
            result[f"max_vio_per_layer/layer_{layer - 1}"] = relative.max().item()
            result[f"min_vio_per_layer/layer_{layer - 1}"] = relative.min().item()
    return result


def merge_packets(packets):
    """Combine compact rank-local records on the logging rank, including empty PP stages."""
    summaries, loads, scalars = {}, {}, {}
    for packet in packets:
        for key, value in packet["summaries"].items():
            value = torch.tensor(value, dtype=torch.float64)
            summaries[key] = merge_summary(summaries[key], value) if key in summaries else value
        for layer, value in packet["loads"].items():
            value = torch.tensor(value, dtype=torch.float64)
            loads[layer] = loads[layer] + value if layer in loads else value
        for key, value in packet["scalars"].items():
            value = torch.tensor(value, dtype=torch.float64)
            scalars[key] = scalars[key] + value if key in scalars else value
    result = statistic_tags(summaries)
    result.update(load_balance_tags(loads))
    totals = {}
    for (name, layer), value in scalars.items():
        result[f"load_balance/{name}/layer_{layer - 1}"] = (value[0] / value[1]).item()
        totals[name] = totals[name] + value if name in totals else value
    for name, value in totals.items():
        result[f"load_balance/{name}"] = (value[0] / value[1]).item()
    return result
