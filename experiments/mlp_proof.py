import copy
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import get_dataloaders
from src.metrics import mse_psnr_ssim
from src.models.autoencoder import VanillaAE
from src.utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLF_EPOCHS = 30
DEC_EPOCHS = 20


class MLPClassifier(nn.Module):
    """784 -> 256 -> 64 -> 10 (ReLU), trained with cross-entropy."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def features(self, x):  # z_MLP (64-d), the layer before the final classifier
        x = x.view(x.size(0), -1)
        return F.relu(self.fc2(F.relu(self.fc1(x))))

    def forward(self, x):
        return self.fc3(self.features(x))


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def train_classifier(model, train_loader, val_loader):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc, best_state = 0.0, None
    for ep in range(1, CLF_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_acc = accuracy(model, val_loader)
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
        print("[clf] epoch %02d  loss %.4f  val acc %.4f" % (ep, loss.item(), val_acc))
    model.load_state_dict(best_state)
    return model


def train_decoder_on_frozen(decoder, feat_fn, train_loader):
    opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    for ep in range(1, DEC_EPOCHS + 1):
        decoder.train()
        total, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            with torch.no_grad():
                z = feat_fn(x)
            loss = F.mse_loss(decoder(z), x.view(x.size(0), -1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        print("[frozen-dec] epoch %02d  mse %.5f" % (ep, total / n))
    return decoder


def train_ae(ae, train_loader):
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    for ep in range(1, DEC_EPOCHS + 1):
        ae.train()
        total, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            loss = F.mse_loss(ae(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        print("[vanilla-ae] epoch %02d  mse %.5f" % (ep, total / n))
    return ae


def main():
    set_seed(42)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128, scale="01")

    # ---------- Stage 1: train MLP classifier ----------
    mlp = train_classifier(MLPClassifier().to(DEVICE), train_loader, val_loader)
    test_acc = accuracy(mlp, test_loader)
    print("\nSTAGE 1: MLP test accuracy = %.4f (need >= 0.88)" % test_acc)
    if test_acc < 0.88:
        print("WARNING: below 88%. Increase CLF_EPOCHS and rerun.")
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad = False  # freeze for Stage 4

    # PyTorch weights are (out, in): W1 (256,784), W2 (64,256), W3 (10,64).
    # The brief's W^T h  ==  h @ W in PyTorch (no explicit transpose needed).
    W1, W2, W3 = mlp.fc1.weight, mlp.fc2.weight, mlp.fc3.weight

    # ---------- Stage 2: top-down projection from one-hot class ----------
    y_onehot = torch.eye(10, device=DEVICE)
    z_hat = F.relu(y_onehot @ W3)
    h1_hat = F.relu(z_hat @ W2)
    x_class = torch.sigmoid(h1_hat @ W1).view(-1, 1, 28, 28)

    # ---------- Stage 3: latent transpose projection ----------
    def transpose_recon(x):
        z = mlp.features(x)
        return torch.sigmoid(F.relu(z @ W2) @ W1).view(-1, 1, 28, 28)

    # ---------- Stage 4: frozen latent decoder vs Vanilla AE ----------
    frozen_dec = nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid()
    ).to(DEVICE)
    frozen_dec = train_decoder_on_frozen(frozen_dec, mlp.features, train_loader)

    def frozen_recon(x):
        return frozen_dec(mlp.features(x)).view(-1, 1, 28, 28)

    set_seed(42)
    ae = train_ae(VanillaAE(64).to(DEVICE), train_loader)
    ae.eval()
    frozen_dec.eval()

    results = {"mlp_test_acc": test_acc}
    for name, fn in [("transpose_projection", transpose_recon),
                     ("frozen_latent_decoder", frozen_recon),
                     ("vanilla_ae_64d", lambda x: ae(x))]:
        mse, psnr, ssim = mse_psnr_ssim(fn, test_loader, DEVICE)
        results[name] = {"mse": mse, "psnr": psnr, "ssim": ssim}
        print("STAGE 3/4  %-24s MSE %.5f  PSNR %.2f dB  SSIM %.4f" % (name, mse, psnr, ssim))

    # ---------- Stage 5: prior sampling in z_MLP space ----------
    with torch.no_grad():
        zs = torch.cat([mlp.features(x.to(DEVICE)) for x, _ in test_loader])
        mu, sigma = zs.mean(0), zs.std(0)
        z_rand = mu + sigma * torch.randn(10, 64, device=DEVICE)
        x_rand = frozen_dec(z_rand).view(-1, 1, 28, 28)

    # ---------- Figure 3 ----------
    # one test image per class, so columns line up with one-hot class projections
    picked = {}
    for x, y in test_loader:
        for xi, yi in zip(x, y):
            if yi.item() not in picked:
                picked[yi.item()] = xi
        if len(picked) == 10:
            break
    originals = torch.stack([picked[c] for c in range(10)]).to(DEVICE)

    with torch.no_grad():
        rows = [
            ("(1) Original", originals),
            ("(2a) W^T from one-hot class", x_class),
            ("(2b) W^T from z_MLP", transpose_recon(originals)),
            ("(3a) Frozen decoder (z_MLP)", frozen_recon(originals)),
            ("(3b) Vanilla AE (z_AE)", ae(originals)),
            ("(4) Prior sampling N(mu,sigma^2)", x_rand),
        ]

    fig, axes = plt.subplots(len(rows), 10, figsize=(15, 9.5))
    for r, (label, imgs) in enumerate(rows):
        imgs = imgs.detach().cpu().clamp(0, 1)
        for c in range(10):
            ax = axes[r, c]
            ax.imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.subplots_adjust(left=0.2, wspace=0.05, hspace=0.08)
    fig.suptitle("Figure 3: Why an MLP classifier cannot generate images", fontsize=13)
    fig.savefig("outputs/figures/fig3_mlp_proof.png", dpi=200, bbox_inches="tight")

    with open("outputs/logs/mlp_proof.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved outputs/figures/fig3_mlp_proof.png and outputs/logs/mlp_proof.json")


if __name__ == "__main__":
    main()