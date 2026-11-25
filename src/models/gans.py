import torch.nn as nn

NZ = 100


class GanG(nn.Module):  # Vanilla GAN generator (MLP)
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NZ, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1)).view(-1, 1, 28, 28)


class GanD(nn.Module):  # Vanilla GAN discriminator (MLP), outputs logits
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class DcganG(nn.Module):  # 100 -> 7x7 -> 14x14 -> 28x28
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(NZ, 256, 7, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), NZ, 1, 1))


class DcganD(nn.Module):  # strided convs, no pooling
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def dcgan_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)