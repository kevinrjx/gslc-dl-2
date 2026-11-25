import torch.nn as nn


class VanillaAE(nn.Module):
    """784 -> 256 -> latent -> 256 -> 784, same shape as the MLP classifier's feature extractor."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x.view(x.size(0), -1))
        return self.decoder(z).view(-1, 1, 28, 28)