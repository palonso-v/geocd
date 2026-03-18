import torch

# Adjust these imports to your actual project structure
from utils import PointCloudAE, PointCloudTransformerAE_fullmodel
from PTV3.model import PointTransformerV3


def build_model(backbone, latent_dim=128, device="cuda"):
    if backbone == "AE":
        model = PointCloudAE(
            2048,
            latent_size=latent_dim,
        )

    elif backbone == "PTv3":
        encoder = PointTransformerV3(
            in_channels=3,
            cls_mode=False,
            enable_flash=False,
        )

        model = PointCloudTransformerAE_fullmodel(encoder)

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model.to(device)


def make_ptv3_input(batch: torch.Tensor, grid_size: float = 0.01) -> dict:
    """
    Convert a batch of point clouds from shape (B, N, 3)
    into PointTransformerV3 / Pointcept-style input format.
    """
    batch = batch.contiguous()
    b, n, c = batch.shape

    pointcloud = batch.view(b * n, c)
    batch_indices = (
        torch.arange(b, device=batch.device)
        .unsqueeze(1)
        .expand(b, n)
        .reshape(-1)
    )

    return {
        "coord": pointcloud,
        "batch": batch_indices,
        "feat": pointcloud,
        "grid_size": torch.tensor(grid_size, device=batch.device),
    }