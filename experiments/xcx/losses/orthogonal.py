"""YuLan's differentiable row-correlation objective, including optional PCA removal."""

import torch
import torch.nn.functional as F


def router_orthogonal_loss(rows, *, preprocess="none", pca_rank=1):
    weights = rows.float()
    if preprocess != "none":
        weights = weights - weights.mean(dim=0, keepdim=True)
        if preprocess == "pca":
            rank = min(pca_rank, *weights.shape)
            if rank > 0 and weights.numel() > 0:
                _, _, vh = torch.linalg.svd(weights, full_matrices=False)
                components = vh[:rank]
                weights = weights - (weights @ components.T) @ components
        elif preprocess != "mean_center":
            raise ValueError("Unknown orthogonal preprocessing")
    weights = F.normalize(weights, p=2, dim=-1, eps=1e-8)
    gram = weights @ weights.T
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return ((gram - identity) ** 2).mean()
