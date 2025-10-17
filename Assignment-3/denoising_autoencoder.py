import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class DenoiseAE(nn.Module):
    def __init__(self, latent_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z
    

def add_noise(x, noise_level):
    noise = torch.randn_like(x)*noise_level
    xn = x + noise
    return torch.clamp(xn, 0.0, 1.0)


def train_epoch(model, loader, optim, criterion, device, noise_level):
    model.train()
    total_loss = 0.0
    for xb, _ in loader:
        xb = xb.view(xb.size(0), -1).to(device)
        xb_noisy = add_noise(xb, noise_level)
        optim.zero_grad()
        out, _ = model(xb_noisy)
        loss = criterion(out, xb)
        loss.backward()
        optim.step()
        total_loss += loss.item()*xb.size(0)
    return total_loss / len(loader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.5, help="Noise stddev")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = os.path.join("outputs", "part4", f"latent{args.latent_dim}_noise{args.noise}")
    ensure_dir(outdir)

    transform = transforms.ToTensor()
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(testset,  batch_size=256, shuffle=False, num_workers=2)

    model = DenoiseAE(args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, args.noise)
        print(f"Epoch {epoch}/{args.epochs} - loss {loss:.6f}")

    model.eval()
    noise_levels = [0.3, 0.5, 0.8, 0.9]
    with torch.no_grad():
        xb, _ = next(iter(test_loader))
        xb_flat = xb.view(xb.size(0), -1).to(device)
        for nl in noise_levels:
            xb_noisy = add_noise(xb_flat, nl)
            out, _ = model(xb_noisy)
            out_img = out.cpu().view(-1,1,28,28)
            noisy_img = xb_noisy.cpu().view(-1,1,28,28)
            save_path = os.path.join(outdir, f"recon_noise_{nl:.2f}.png")
            utils.save_image(torch.cat([xb[:64], noisy_img[:64], out_img[:64]]), save_path, nrow=8, normalize=True)
            print("Saved", save_path)