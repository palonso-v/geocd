import os
import torch
import torch.optim as optim

from models import build_model, make_ptv3_input
from losses import chamfer_loss, geocd_loss
from utils import set_bn_eval
from data_utils import load_data

def forward_model(model, batch, backbone):
    if backbone == "AE":
        return model(batch)
    elif backbone == "PTv3":
        return model(make_ptv3_input(batch))
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def compute_loss(recon, batch, stage, config):
    if stage == "cd":
        return chamfer_loss(recon, batch)
    elif stage == "geocd":
        return geocd_loss(
            recon,
            batch,
            k=config.k_value,
            n_hops=config.nhops,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")


def get_default_lr(backbone, stage):
    if backbone == "AE":
        return 5e-4 if stage == "cd" else 1e-4
    elif backbone == "PTv3":
        return 1e-4 if stage == "cd" else 5e-5
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def validate(model, val_loader, device, config, stage):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            recon = forward_model(model, batch, config.backbone)
            loss = compute_loss(recon, batch, stage, config)
            losses.append(loss.item())

    return sum(losses) / len(losses)


def save_checkpoint(model, optimizer, epoch, val_loss, best_val_loss, config, stage):
    os.makedirs(config.save_dir, exist_ok=True)

    prefix = f"geocd_{config.backbone}_{config.dataset}_{stage}"

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path = os.path.join(config.save_dir, f"{prefix}_best.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "stage": stage,
            },
            best_path,
        )
        print(f"✅ Best checkpoint saved: {best_path} | val loss: {val_loss:.6f}")

    last_path = os.path.join(config.save_dir, f"{prefix}_last.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "stage": stage,
        },
        last_path,
    )

    return best_val_loss


def run_training(config, stage):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    _, train_loader, _, val_loader = load_data(
        config.dataset,
        config.batch_size,
    )

    model = build_model(
        backbone=config.backbone,
        latent_dim=config.latent_dim,
        device=device,
    )

    lr = config.lr if config.lr is not None else get_default_lr(config.backbone, stage)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Dataset: {config.dataset}")
    print(f"Backbone: {config.backbone}")
    print(f"Stage: {stage}")
    print(f"Learning rate: {lr}")

    start_epoch = 0

    if stage == "geocd":
        if not os.path.exists(config.checkpoint_path):
            raise ValueError("GeoCD finetuning requires a valid checkpoint_path.")

        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from: {config.checkpoint_path}")

    elif stage == "cd" and getattr(config, "load_checkpoint", False):
        if not config.checkpoint_path:
            raise ValueError("load_checkpoint was set, but checkpoint_path is empty.")
        if not os.path.exists(config.checkpoint_path):
            raise ValueError(f"Checkpoint not found: {config.checkpoint_path}")

        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}")

    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.nepochs):
        model.train()

        if stage == "geocd" and config.batch_size == 1:
            model.apply(set_bn_eval)

        total_loss = 0.0

        for batch, _ in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon = forward_model(model, batch, config.backbone)
            loss = compute_loss(recon, batch, stage, config)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = validate(model, val_loader, device, config, stage)

        print(
            f"Epoch {epoch + 1}/{config.nepochs} | "
            f"train loss: {train_loss:.6f} | val loss: {val_loss:.6f}"
        )

        best_val_loss = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            config=config,
            stage=stage,
        )

    return 0


def evaluate_loader(model, loader, device, config, metrics):
    model.eval()

    metric_sums = {metric: 0.0 for metric in metrics}
    num_batches = 0

    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device)
            recon = forward_model(model, batch, config.backbone)

            if "cd" in metrics:
                metric_sums["cd"] += chamfer_loss(recon, batch).item()

            if "geocd" in metrics:
                metric_sums["geocd"] += geocd_loss_eval(
                    recon,
                    batch,
                    k=config.k_value,
                    n_hops=config.nhops,
                ).item()

            num_batches += 1

    if num_batches == 0:
        raise ValueError("Evaluation loader is empty.")

    metric_means = {k: v / num_batches for k, v in metric_sums.items()}
    return metric_means


def evaluate_reconstruction(config):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    _, train_loader, _, val_loader = load_data(
        config.dataset,
        config.batch_size,
    )

    if config.split == "train":
        loader = train_loader
    elif config.split == "val":
        loader = val_loader
    else:
        raise ValueError(f"Unknown split: {config.split}")

    model = build_model(
        backbone=config.backbone,
        latent_dim=config.latent_dim,
        device=device,
    )

    if not os.path.exists(config.checkpoint_path):
        raise ValueError(f"Checkpoint not found: {config.checkpoint_path}")

    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from: {config.checkpoint_path}")
    print(f"Dataset: {config.dataset}")
    print(f"Backbone: {config.backbone}")
    print(f"Split: {config.split}")
    print(f"Metrics: {config.metrics}")

    results = evaluate_loader(
        model=model,
        loader=loader,
        device=device,
        config=config,
        metrics=config.metrics,
    )

    print("\nEvaluation results:")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value:.6f}")

    return results


def train_cd(config):
    return run_training(config, stage="cd")


def finetune_geocd(config):
    return run_training(config, stage="geocd")