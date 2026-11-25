import json, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from src.data import get_dataloaders
from src.models.autoencoder import VanillaAE
from src.models.ae_family import SparseAE, VAE
from src.models.gans import GanG, DcganG, NZ
from src.models.ddpm import DDPM
from src.utils import set_seed

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(m, name):
    m.load_state_dict(torch.load("outputs/checkpoints/" + name, map_location=dev))
    return m.to(dev).eval()


def show(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


@torch.no_grad()
def main():
    set_seed(42)
    ae = load(VanillaAE(64), "vanilla_ae.pt")
    sp = load(SparseAE(64), "sparse_ae.pt")
    vae = load(VAE(64), "vae.pt")
    gan = load(GanG(), "gan_G.pt")
    dcgan = load(DcganG(), "dcgan_G.pt")
    ddpm = load(DDPM(100), "ddpm.pt")

    # ---------- Figure 2: 4x4 random samples ----------
    set_seed(42)
    samples = [
        ("A: VAE", vae.sample(16)),
        ("B: Vanilla GAN", (gan(torch.randn(16, NZ, device=dev)) + 1) / 2),
        ("C: DCGAN", (dcgan(torch.randn(16, NZ, device=dev)) + 1) / 2),
        ("D: DDPM", (ddpm.sample(16) + 1) / 2),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (t, s) in zip(axes, samples):
        g = make_grid(s.cpu(), nrow=4, padding=2, pad_value=1)
        ax.imshow(g.permute(1, 2, 0)); ax.set_title(t); ax.axis("off")
    fig.suptitle("Figure 2: Random generative samples (4x4)")
    fig.savefig("outputs/figures/fig2_random_sampling.png", dpi=200, bbox_inches="tight")

    # ---------- Figure 4: linear latent interpolation ----------
    _, _, test = get_dataloaders(128, "01")
    xa = xb = None
    for x, y in test:
        for xi, yi in zip(x, y):
            if yi.item() == 1 and xa is None: xa = xi
            if yi.item() == 8 and xb is None: xb = xi
        if xa is not None and xb is not None:
            break
    xa, xb = xa.to(dev).unsqueeze(0), xb.to(dev).unsqueeze(0)
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    enc = {
        "Vanilla AE": lambda x: ae.encoder(x.view(1, -1)),
        "Sparse AE": lambda x: sp.enc2(sp.enc1(x.view(1, -1))),
        "VAE": lambda x: vae.mu(vae.enc(x.view(1, -1))),
    }
    dec = {"Vanilla AE": ae.decoder, "Sparse AE": sp.decoder, "VAE": vae.decoder}
    rows = {}
    for name in enc:
        za, zb = enc[name](xa), enc[name](xb)
        rows[name] = [dec[name]((1 - a) * za + a * zb).view(28, 28).cpu() for a in alphas]
    set_seed(42)
    for name, G in [("Vanilla GAN", gan), ("DCGAN", dcgan)]:
        za, zb = torch.randn(1, NZ, device=dev), torch.randn(1, NZ, device=dev)
        rows[name] = [((G((1 - a) * za + a * zb) + 1) / 2).view(28, 28).cpu() for a in alphas]

    order = ["VAE", "DCGAN", "Vanilla GAN", "Sparse AE", "Vanilla AE"]
    fig, axes = plt.subplots(5, 6, figsize=(9, 8))
    for r, name in enumerate(order):
        for c, a in enumerate(alphas):
            show(axes[r, c], rows[name][c], "a=%.1f" % a if r == 0 else None)
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle("Figure 4: Linear latent interpolation (AE/VAE: trouser to bag; GANs: random z)")
    fig.savefig("outputs/figures/fig4_interpolation.png", dpi=200, bbox_inches="tight")

    # ---------- Figure 5: loss curves + DDPM trajectory ----------
    a = json.load(open("outputs/logs/ae_family.json"))
    g = json.load(open("outputs/logs/gans.json"))
    d = json.load(open("outputs/logs/ddpm.json"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, k, t in [(axes[0, 0], "vanilla_ae", "Vanilla AE (MSE)"),
                     (axes[0, 1], "sparse_ae", "Sparse AE (MSE + L1)"),
                     (axes[0, 2], "vae", "VAE (neg. ELBO)")]:
        ax.plot(a[k]["history"]["train"], label="train")
        ax.plot(a[k]["history"]["val"], label="val")
        ax.set_title(t); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    for ax, k, t in [(axes[1, 0], "gan", "Vanilla GAN"), (axes[1, 1], "dcgan", "DCGAN")]:
        ax.plot(g[k]["history"]["G"], label="Generator")
        ax.plot(g[k]["history"]["D"], label="Discriminator")
        ax.set_title(t); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    axes[1, 2].plot(d["history"]["loss"])
    axes[1, 2].set_title("DDPM (noise-prediction MSE)")
    axes[1, 2].set_xlabel("epoch"); axes[1, 2].set_ylabel("loss")
    fig.suptitle("Figure 5a: Training loss curves")
    fig.savefig("outputs/figures/fig5_loss_curves.png", dpi=200, bbox_inches="tight")

    traj = torch.load("outputs/logs/ddpm_traj.pt")
    n = len(traj)
    fig, axes = plt.subplots(4, n, figsize=(1.2 * n, 5))
    for r in range(4):
        for c in range(n):
            show(axes[r, c], ((traj[c][r, 0] + 1) / 2).clamp(0, 1),
                 ("x_T" if c == 0 else ("x_0" if c == n - 1 else None)) if r == 0 else None)
    fig.suptitle("Figure 5b: DDPM reverse denoising trajectory (x_T to x_0)")
    fig.savefig("outputs/figures/fig5_ddpm_trajectory.png", dpi=200, bbox_inches="tight")
    print("Saved Figures 2, 4, 5")


if __name__ == "__main__":
    main()