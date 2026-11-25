import torch
import torch.nn as nn


class SparseAE(nn.Module):
    """784 -> 256 (L1-penalised hidden h) -> 64 -> 256 -> 784."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
        self.enc2 = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.enc1(x.view(x.size(0), -1))
        return self.decoder(self.enc2(h)).view(-1, 1, 28, 28), h


class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.enc(x.view(x.size(0), -1))
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # reparameterization
        return self.decoder(z).view(-1, 1, 28, 28), mu, logvar

    def reconstruct(self, x):  # deterministic (uses mu) for evaluation
        h = self.enc(x.view(x.size(0), -1))
        return self.decoder(self.mu(h)).view(-1, 1, 28, 28)

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.latent_dim, device=next(self.parameters()).device)
        return self.decoder(z).view(-1, 1, 28, 28)