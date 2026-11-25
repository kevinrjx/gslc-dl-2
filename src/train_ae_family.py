import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.data import get_dataloaders
from src.metrics import mse_psnr_ssim, measure_latency, compute_fid
from src.models.autoencoder import VanillaAE
from src.models.ae_family import SparseAE, VAE
from src.utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
LAMBDA_SPARSE = 1e-5  # weight of the L1 penalty on the 256-d hidden layer (tune if needed)


def loss_fn(name, model, x):
    if name == "vanilla_ae":
        return F.mse_loss(model(x), x)
    if name == "sparse_ae":
        r, h = model(x)
        return F.mse_loss(r, x) + LAMBDA_SPARSE * h.abs().sum(1).mean()
    r, mu, lv = model(x)  # VAE: negative ELBO = BCE (sum over pixels) + KL
    bce = F.binary_cross_entropy(r, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
    return bce + kl


def recon_fn(name, model):
    if name == "vanilla_ae":
        return lambda x: model(x)
    if name == "sparse_ae":
        return lambda x: model(x)[0]
    return model.reconstruct


def train(name, model, train_loader, val_loader):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = {"train": [], "val": [], "epoch_time": []}
    for ep in range(1, EPOCHS + 1):
        model.train()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tot, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            loss = loss_fn(name, model, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        hist["epoch_time"].append(time.perf_counter() - t0)
        model.eval()
        with torch.no_grad():
            vt = sum(loss_fn(name, model, x.to(DEVICE)).item() * x.size(0) for x, _ in val_loader)
        hist["train"].append(tot / n)
        hist["val"].append(vt / len(val_loader.dataset))
        print("[%s] epoch %02d  train %.5f  val %.5f  (%.1fs)" % (
            name, ep, hist["train"][-1], hist["val"][-1], hist["epoch_time"][-1]))
    return hist


def main():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    train_loader, val_loader, test_loader = get_dataloaders(128, "01")
    x64 = next(iter(test_loader))[0][:64].to(DEVICE)

    models = {"vanilla_ae": VanillaAE, "sparse_ae": SparseAE, "vae": VAE}
    results, recons = {}, {}
    x8 = next(iter(test_loader))[0][:8].to(DEVICE)

    for name, cls in models.items():
        set_seed(42)
        model = cls(64).to(DEVICE)
        hist = train(name, model, train_loader, val_loader)
        model.eval()
        fn = recon_fn(name, model)
        mse, psnr, ssim = mse_psnr_ssim(fn, test_loader, DEVICE)
        lat = measure_latency(lambda: fn(x64), DEVICE)
        r = {"latent_dim": 64, "time_per_epoch_s": sum(hist["epoch_time"]) / EPOCHS,
             "latency_ms_batch64": lat, "test_mse": mse, "test_psnr": psnr,
             "test_ssim": ssim, "history": hist}
        if name == "vae":
            set_seed(42)
            r["fid"] = compute_fid(test_loader, model.sample, DEVICE, n_samples=5000)
            torch.save(model.state_dict(), "outputs/checkpoints/vae.pt")
        else:
            torch.save(model.state_dict(), "outputs/checkpoints/%s.pt" % name)
        results[name] = r
        print("TEST %s: MSE %.5f  PSNR %.2f  SSIM %.4f  latency %.2f ms  %s" % (
            name, mse, psnr, ssim, lat, ("FID %.2f" % r["fid"]) if "fid" in r else ""))
        with torch.no_grad():
            recons[name] = fn(x8).cpu()

    # Figure 1: originals + 3 reconstructions
    rows = [("Original", x8.cpu()), ("Vanilla AE", recons["vanilla_ae"]),
            ("Sparse AE", recons["sparse_ae"]), ("VAE", recons["vae"])]
    fig, axes = plt.subplots(4, 8, figsize=(12, 6.5))
    for r_i, (label, imgs) in enumerate(rows):
        for c in range(8):
            ax = axes[r_i, c]
            ax.imshow(imgs[c, 0].clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle("Figure 1: Reconstruction comparison (test set)")
    fig.savefig("outputs/figures/fig1_reconstruction.png", dpi=200, bbox_inches="tight")

    with open("outputs/logs/ae_family.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved fig1 and outputs/logs/ae_family.json")


if __name__ == "__main__":
    main()