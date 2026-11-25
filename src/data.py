import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms

from src.utils import set_seed


def get_dataloaders(batch_size=128, scale="01", data_dir="data", seed=42):
    """
    Fashion-MNIST, merged (70k) then split 80/10/10.
    scale="01"  -> pixels in [0, 1]   (AE, Sparse AE, VAE, MLP)
    scale="pm1" -> pixels in [-1, 1]  (GAN, DCGAN, DDPM)
    """
    tfms = [transforms.ToTensor()]
    if scale == "pm1":
        tfms.append(transforms.Normalize((0.5,), (0.5,)))
    tfm = transforms.Compose(tfms)

    train_full = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
    test_full = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm)
    full = ConcatDataset([train_full, test_full])  # 70,000 images

    n = len(full)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(seed)  # same split for every model
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    set_seed(42)
    tr, va, te = get_dataloaders()
    print("train/val/test sizes:", len(tr.dataset), len(va.dataset), len(te.dataset))
    x, y = next(iter(tr))
    print("batch:", x.shape, "min/max:", x.min().item(), x.max().item())