import torch
from external.chamfer_distance import ChamferDistance as cd_cuda


def chamfer_loss(recon, target):
    dist1, dist2 = cd_cuda()(recon, target)
    return torch.mean(dist1) + torch.mean(dist2)


def pairwise_distances(x):
    """
    x: (B, N, D)
    returns: (B, N, N)
    """
    x_inner = -2 * torch.bmm(x, x.transpose(1, 2))
    x_square = torch.sum(x ** 2, dim=-1, keepdim=True)
    dist = x_square + x_inner + x_square.transpose(1, 2)
    dist = torch.clamp(dist, min=0.0)
    return torch.sqrt(dist + 1e-8)


def differentiable_softmin(x, dim=-1):
    return -torch.logsumexp(-x, dim=dim)


def geocd_loss(x, y, k=10, n_hops=3):

    b, n, _ = x.shape
    m = y.shape[1]
    device = x.device

    merged = torch.cat([x, y], dim=1)
    total_points = merged.shape[1]

    pairwise_dists = pairwise_distances(merged)

    knn_dists, knn_idx = torch.topk(pairwise_dists, k=k + 1, largest=False)
    knn_dists = knn_dists[:, :, 1:]
    knn_idx = knn_idx[:, :, 1:]

    geo_dists = torch.full_like(pairwise_dists, 1e6)

    batch_indices = torch.arange(b, device=device).view(b, 1, 1)
    point_indices = torch.arange(total_points, device=device).view(1, total_points, 1).expand(b, -1, k)

    geo_dists[batch_indices, point_indices, knn_idx] = knn_dists
    geo_dists[batch_indices, knn_idx, point_indices] = knn_dists

    for _ in range(n_hops - 1):
        expanded_dists = torch.bmm(geo_dists, geo_dists)
        geo_dists = torch.min(geo_dists, expanded_dists)

    x_idx = torch.arange(n, device=device)
    y_idx = torch.arange(n, n + m, device=device)

    d_x_to_y = geo_dists[:, x_idx][:, :, y_idx]
    d_y_to_x = geo_dists[:, y_idx][:, :, x_idx]

    d_x_to_y_softmin = differentiable_softmin(d_x_to_y, dim=-1)
    d_y_to_x_softmin = differentiable_softmin(d_y_to_x, dim=-1)

    return (d_x_to_y_softmin.mean(dim=-1) + d_y_to_x_softmin.mean(dim=-1)).mean()


def geocd_loss_eval(x, y, k=10, n_hops=3):

    b, n, _ = x.shape
    m = y.shape[1]
    device = x.device

    merged = torch.cat([x, y], dim=1)
    total_points = merged.shape[1]

    with torch.no_grad():
        pairwise_dists = pairwise_distances(merged)

        knn_dists, knn_idx = torch.topk(pairwise_dists, k=k + 1, largest=False)
        knn_dists = knn_dists[:, :, 1:]
        knn_idx = knn_idx[:, :, 1:]

        geo_dists = torch.full_like(pairwise_dists, 1e6)

        batch_indices = torch.arange(b, device=device).view(b, 1, 1)
        point_indices = torch.arange(total_points, device=device).view(1, total_points, 1).expand(b, -1, k)

        geo_dists[batch_indices, point_indices, knn_idx] = knn_dists
        geo_dists[batch_indices, knn_idx, point_indices] = knn_dists

        for _ in range(n_hops - 1):
            expanded_dists = torch.bmm(geo_dists, geo_dists)
            geo_dists = torch.min(geo_dists, expanded_dists)

    x_idx = torch.arange(n, device=device)
    y_idx = torch.arange(n, n + m, device=device)

    d_x_to_y = geo_dists[:, x_idx][:, :, y_idx]
    d_y_to_x = geo_dists[:, y_idx][:, :, x_idx]

    d_x_to_y_softmin = differentiable_softmin(d_x_to_y, dim=-1)
    d_y_to_x_softmin = differentiable_softmin(d_y_to_x, dim=-1)

    return (d_x_to_y_softmin.mean(dim=-1) + d_y_to_x_softmin.mean(dim=-1)).mean()