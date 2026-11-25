import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def temb_fn(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.t = nn.Linear(tdim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.t(e)[:, :, None, None]
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, b=32, tdim=64):
        super().__init__()
        self.tdim = tdim
        self.tm = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inc = nn.Conv2d(1, b, 3, padding=1)
        self.d1 = Block(b, b, tdim)
        self.d2 = Block(b, 2 * b, tdim)
        self.u1 = Block(3 * b, b, tdim)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x, t):
        e = self.tm(temb_fn(t, self.tdim))
        h1 = self.d1(self.inc(x), e)
        h2 = self.d2(F.avg_pool2d(h1, 2), e)
        u = self.u1(torch.cat([F.interpolate(h2, scale_factor=2), h1], 1), e)
        return self.out(u)


class DDPM(nn.Module):
    def __init__(self, T=100):
        super().__init__()
        self.T = T
        self.net = UNet()
        betas = torch.linspace(1e-4, 0.02 * 1000 / T, T)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("abar", torch.cumprod(1 - betas, 0))

    def loss(self, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        eps = torch.randn_like(x0)
        ab = self.abar[t][:, None, None, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return F.mse_loss(self.net(xt, t), eps)

    @torch.no_grad()
    def sample(self, n, return_traj=False, traj_steps=10):
        dev = self.betas.device
        x = torch.randn(n, 1, 28, 28, device=dev)
        traj = [x[:8].cpu()] if return_traj else None
        every = max(1, self.T // traj_steps)
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=dev, dtype=torch.long)
            eps = self.net(x, tt)
            a, ab, b = self.alphas[t], self.abar[t], self.betas[t]
            x = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if t > 0:
                x = x + b.sqrt() * torch.randn_like(x)
            if return_traj and t % every == 0:
                traj.append(x[:8].cpu())
        x = x.clamp(-1, 1)
        return (x, traj) if return_traj else x