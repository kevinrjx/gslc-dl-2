import json, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from src.data import get_dataloaders
from src.metrics import measure_latency, compute_fid
from src.models.ddpm import DDPM
from src.utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, T = 8, 100


def main():
    for d in ["outputs/figures", "outputs/logs", "outputs/checkpoints"]:
        os.makedirs(d, exist_ok=True)
    set_seed(42)
    train_loader, _, _ = get_dataloaders(256, "pm1")
    _, _, test01 = get_dataloaders(128, "01")

    model = DDPM(T).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    hist = {"loss": [], "epoch_time": []}
    for ep in range(1, EPOCHS + 1):
        model.train()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tot, n = 0.0, 0
        for x, _ in train_loader:
            loss = model.loss(x.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        hist["loss"].append(tot / n)
        hist["epoch_time"].append(time.perf_counter() - t0)
        print("[ddpm] epoch %02d  loss %.5f  (%.1fs)" % (ep, hist["loss"][-1], hist["epoch_time"][-1]))
    torch.save(model.state_dict(), "outputs/checkpoints/ddpm.pt")

    model.eval()
    lat = measure_latency(lambda: model.sample(64), DEVICE, warmup=1, runs=3)
    sample01 = lambda n: (model.sample(n) + 1) / 2
    set_seed(42)
    fid = compute_fid(test01, sample01, DEVICE, n_samples=5000)
    print("TEST ddpm: latency %.1f ms  FID %.2f" % (lat, fid))

    set_seed(42)
    grid = make_grid(sample01(16).cpu(), nrow=4, padding=2, pad_value=1)
    plt.figure(figsize=(4.5, 4.5)); plt.imshow(grid.permute(1, 2, 0)); plt.axis("off")
    plt.title("DDPM samples"); plt.savefig("outputs/figures/fig2_ddpm_panel_D.png", dpi=200, bbox_inches="tight")

    set_seed(42)
    _, traj = model.sample(8, return_traj=True, traj_steps=10)
    torch.save(traj, "outputs/logs/ddpm_traj.pt")
    with open("outputs/logs/ddpm.json", "w") as f:
        json.dump({"latent_dim": "28x28", "T": T, "time_per_epoch_s": sum(hist["epoch_time"]) / EPOCHS,
                   "latency_ms_batch64": lat, "fid": fid, "history": hist}, f, indent=2)
    print("Saved DDPM results")


if __name__ == "__main__":
    main()