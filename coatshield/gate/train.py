"""Train the CNN gate on synthetic silhouettes (GPU recommended; not needed for the baseline).

Run as a module:  python -m coatshield.gate.train
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn

from coatshield.config import REPO_ROOT, Config, load_config
from coatshield.gate.cnn import GateCNN, fit_temperature
from coatshield.gate.silhouettes import make_set
from coatshield.seeds import set_torch_determinism

CHECKPOINT = REPO_ROOT / "models" / "checkpoints" / "gate_cnn.pt"


def to_tensor(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images.astype(np.float32) / 255.0)[:, None]


def train_gate(cfg: Config, n_train: int, n_val: int, epochs: int, device: str, log=print):
    """AdamW + cross-entropy, then temperature scaling on the validation images."""
    gv = cfg.gate_vision
    set_torch_determinism(cfg.seed, warn_only=True)
    x, y, _ = make_set(cfg, "train", n_train)
    xv, yv, _ = make_set(cfg, "val", n_val)
    x, y = to_tensor(x), torch.from_numpy(y)
    xv, yv = to_tensor(xv).to(device), torch.from_numpy(yv).to(device)
    model = GateCNN(gv.cnn_channels).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=gv.cnn_lr)
    order = torch.Generator().manual_seed(cfg.seed)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x), generator=order)
        losses = []
        for i in range(0, len(x), gv.cnn_batch_size):
            idx = perm[i : i + gv.cnn_batch_size]
            flip = torch.rand((), generator=order) < 0.5  # mirror: shape classes are symmetric
            batch = x[idx].flip(-1) if flip else x[idx]
            optim.zero_grad()
            loss = nn.functional.cross_entropy(model(batch.to(device)), y[idx].to(device))
            loss.backward()
            optim.step()
            losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            accuracy = float((model(xv).argmax(1) == yv).float().mean())
        log(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "val_accuracy": accuracy}))
    model.eval()
    with torch.no_grad():
        logits = model(xv)
    temperature = fit_temperature(model, logits.cpu(), yv.cpu())
    return model.cpu(), {"val_accuracy": accuracy, "temperature": temperature,
                         "parameters": sum(p.numel() for p in model.parameters())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config()
    gv = cfg.gate_vision
    model, info = train_gate(cfg, args.n_train or gv.n_images, gv.n_tune,
                             args.epochs or gv.cnn_epochs, args.device)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "channels": gv.cnn_channels, "info": info},
               CHECKPOINT)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
