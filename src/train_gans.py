import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from src.data import get_dataloaders
from src.metrics import measure_latency, compute_fid
from src.models.gans import GanG, GanD, DcganG, DcganD, dcgan_init, NZ
from src.utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BCE = F.binary_cross_entropy_with_logits


def train_gan(name, G, D, loader):
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    hist = {"G": [], "D": [], "epoch_time": []}
    for ep in range(1, EPOCHS + 1):
        G.train(); D.train()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gl = dl = 0.0
        nb = 0
        for x, _ in loader:
            x = x.to(DEVICE)
            b = x.size(0)
            real, fake_lbl = torch.ones(b, 1, device=DEVICE), torch.zeros(b, 1, device=DEVICE)
            # Discriminator step
            z = torch.randn(b, NZ, device=DEVICE)
            fake = G(z).detach()
            lossD = BCE(D(x), real) + BCE(D(fake), fake_lbl)
            optD.zero_grad(); lossD.backward(); optD.step()
            # Generator step (non-saturating loss)
            z = torch.randn(b, NZ, device=DEVICE)
            lossG = BCE(D(G(z)), real)
            optG.zero_grad(); lossG.backward(); optG.step()
            gl += lossG.item(); dl += lossD.item(); nb += 1
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        hist["epoch_time"].append(time.perf_counter() - t0)
        hist["G"].append(gl / nb); hist["D"].append(dl / nb)
        print("[%s] epoch %02d  D %.4f  G %.4f  (%.1fs)" % (
            name, ep, hist["D"][-1], hist["G"][-1], hist["epoch_time"][-1]))
    return hist


def main():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    # GAN models use [-1,1]; FID/metrics use [0,1], and the test split is identical (same seed)
    train_loader, _, _ = get_dataloaders(128, "pm1")
    _, _, test01 = get_dataloaders(128, "01")

    results, grids = {}, {}
    for name, (Gc, Dc) in {"gan": (GanG, GanD), "dcgan": (DcganG, DcganD)}.items():
        set_seed(42)
        G, D = Gc().to(DEVICE), Dc().to(DEVICE)
        if name == "dcgan":
            G.apply(dcgan_init); D.apply(dcgan_init)
        hist = train_gan(name, G, D, train_loader)
        G.eval()

        def sample01(n, G=G):
            with torch.no_grad():
                return (G(torch.randn(n, NZ, device=DEVICE)) + 1) / 2

        z64 = torch.randn(64, NZ, device=DEVICE)
        lat = measure_latency(lambda: G(z64), DEVICE)
        set_seed(42)
        fid = compute_fid(test01, sample01, DEVICE, n_samples=5000)
        results[name] = {"latent_dim": NZ, "time_per_epoch_s": sum(hist["epoch_time"]) / EPOCHS,
                         "latency_ms_batch64": lat, "fid": fid, "history": hist}
        print("TEST %s: latency %.2f ms  FID %.2f" % (name, lat, fid))
        torch.save(G.state_dict(), "outputs/checkpoints/%s_G.pt" % name)
        set_seed(42)
        grids[name] = make_grid(sample01(16).cpu(), nrow=4, padding=2, pad_value=1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    for ax, (name, title) in zip(axes, [("gan", "Vanilla GAN"), ("dcgan", "DCGAN")]):
        ax.imshow(grids[name].permute(1, 2, 0)); ax.set_title(title); ax.axis("off")
    fig.savefig("outputs/figures/fig2_gan_panels_BC.png", dpi=200, bbox_inches="tight")

    with open("outputs/logs/gans.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved outputs/logs/gans.json and GAN sample panels")


if __name__ == "__main__":
    main()