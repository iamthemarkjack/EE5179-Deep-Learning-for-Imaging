import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class AE(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, hidden_size),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z
    

def train_epoch(model, loader, optim, criterion, device, l1_lambda=0.0):
    model.train()
    total = 0.0
    for xb, _ in loader:
        xb = xb.view(xb.size(0), -1).to(device)
        optim.zero_grad()
        out, z = model(xb)
        loss = criterion(out, xb)
        if l1_lambda > 0:
            loss = loss + l1_lambda * z.abs().mean()
        loss.backward()
        optim.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def save_reconstruction_grid(orig, recon, path, nrow=8):
    combined = torch.cat([orig, recon])
    utils.save_image(combined, path, nrow=nrow, normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = os.path.join("outputs", "part2", f"h{args.hidden_size}")
    os.makedirs(outdir, exist_ok=True)

    transform = transforms.ToTensor()
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(testset,  batch_size=256, shuffle=False, num_workers=2)

    model = AE(args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} - loss {loss:.6f}")

    model.eval()
    with torch.no_grad():
        xb, _ = next(iter(test_loader))
        xb_flat = xb.view(xb.size(0), -1).to(device)
        out, z = model(xb_flat)
        out_img = out.cpu().view(-1,1,28,28)
    save_reconstruction_grid(xb[:64], out_img[:64], os.path.join(outdir, "recon.png"), nrow=8)
    print("Saved reconstruction grid to", os.path.join(outdir, "recon.png"))

    weights = model.encoder[0].weight.data.clone().cpu()
    min_w, max_w = weights.min(), weights.max()
    weights_norm = (weights - min_w) / (max_w - min_w + 1e-8)
    weights_img = weights_norm.view(-1,1,28,28)
    utils.save_image(weights_img, os.path.join(outdir, "filters.png"), nrow=8, normalize=False)
    print("Saved filter visualizations to", os.path.join(outdir, "filters.png"))


    random_noise = torch.rand(16,1,28,28)
    with torch.no_grad():
        rn_flat = random_noise.view(16, -1).to(device)
        outn, _ = model(rn_flat)
    utils.save_image(random_noise, os.path.join(outdir, "input_noise.png"), nrow=4, normalize=True)
    utils.save_image(outn.cpu().view(-1,1,28,28), os.path.join(outdir, "recon_noise.png"), nrow=4, normalize=True)
    print("Saved noise input & recon to", outdir)
