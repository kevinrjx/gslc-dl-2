import time
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn


@torch.no_grad()
def mse_psnr_ssim(model_recon_fn, loader, device, data_range=1.0):
    """
    model_recon_fn(x) -> reconstruction, both in [0, 1].
    Returns mean test MSE, PSNR (dB), SSIM over the whole loader.
    """
    mse_sum, psnr_sum, ssim_sum, n = 0.0, 0.0, 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        xr = model_recon_fn(x).clamp(0, 1)
        per_img_mse = F.mse_loss(xr, x, reduction="none").flatten(1).mean(1)
        psnr = 10 * torch.log10(data_range ** 2 / per_img_mse.clamp_min(1e-10))
        xa = x.squeeze(1).cpu().numpy()
        xb = xr.squeeze(1).cpu().numpy()
        for a, b in zip(xa, xb):
            ssim_sum += ssim_fn(a, b, data_range=data_range)
        mse_sum += per_img_mse.sum().item()
        psnr_sum += psnr.sum().item()
        n += x.size(0)
    return mse_sum / n, psnr_sum / n, ssim_sum / n


@torch.no_grad()
def measure_latency(fn, device, warmup=10, runs=50):
    """
    fn() runs ONE batch of 64 generations/reconstructions.
    Returns average milliseconds per batch.
    """
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0


@torch.no_grad()
def compute_fid(real_loader, sample_fn, device, n_samples=5000):
    """
    FID using torchmetrics. Images are expected in [0, 1], shape (B,1,28,28).
    sample_fn(batch_size) -> generated images in [0, 1].
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def prep(x):
        x = x.repeat(1, 3, 1, 1)
        return F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False).clamp(0, 1)

    seen = 0
    for x, _ in real_loader:
        x = x.to(device)
        fid.update(prep(x), real=True)
        seen += x.size(0)
        if seen >= n_samples:
            break

    made = 0
    while made < n_samples:
        b = min(250, n_samples - made)
        fid.update(prep(sample_fn(b).to(device)), real=False)
        made += b
    return fid.compute().item()