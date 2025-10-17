import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class SparseAE(nn.Module):
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
    

def train_epoch(model, loader, optim, criterion, device, l1_lambda=1e-5):
    model.train()
    total_loss = 0.0
    total_act = 0.0
    n = 0
    for xb, _ in loader:
        xb = xb.view(xb.size(0), -1).to(device)
        optim.zero_grad()
        out, z = model(xb)
        recon_loss = criterion(out, xb)
        l1 = z.abs().mean()
        loss = recon_loss + l1_lambda * l1
        loss.backward()
        optim.step()

        total_loss += recon_loss.item() * xb.size(0)
        total_act += l1.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n, total_act / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--l1", type=float, default=1e-4, help="L1 penalty coefficient on hidden activations")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = os.path.join("outputs", "part3", f"h{args.hidden_size}_l1{args.l1}")
    ensure_dir(outdir)

    transform = transforms.ToTensor()
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(testset,  batch_size=256, shuffle=False, num_workers=2)

    model = SparseAE(args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs+1):
        recon_loss, avg_activation = train_epoch(model, train_loader, optimizer, criterion, device, l1_lambda=args.l1)
        print(f"Epoch {epoch}/{args.epochs} - Recon Loss: {recon_loss:.6f} - Avg act (L1): {avg_activation:.6f}")

    model.eval()
    total_act = 0.0
    n = 0
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.view(xb.size(0), -1).to(device)
            _, z = model(xb)
            total_act += z.abs().mean().item() * xb.size(0)
            n += xb.size(0)
    print("Test avg hidden activation (L1 mean):", total_act / n)

    with torch.no_grad():
        xb, _ = next(iter(test_loader))
        xb_flat = xb.view(xb.size(0), -1).to(device)
        out, z = model(xb_flat)
        out_img = out.cpu().view(-1,1,28,28)
    utils.save_image(torch.cat([xb[:64], out_img[:64]]), os.path.join(outdir, "recon.png"), nrow=8, normalize=True)
    print("Saved reconstructions to", os.path.join(outdir, "recon.png"))

    weights = model.encoder[0].weight.data.clone().cpu()
    wmin, wmax = weights.min(), weights.max()
    weights_norm = (weights - wmin) / (wmax - wmin + 1e-8)
    utils.save_image(weights_norm.view(-1,1,28,28), os.path.join(outdir, "filters.png"), nrow=8, normalize=False)
    print("Saved filters to", os.path.join(outdir, "filters.png"))