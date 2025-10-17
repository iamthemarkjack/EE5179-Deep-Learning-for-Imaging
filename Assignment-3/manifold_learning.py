import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class ManifoldAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(True),
            nn.Linear(64, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(True),
            nn.Linear(64, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z
    

def train_epoch(model, loader, optim, criterion, device):
    model.train()
    total = 0.0
    for xb, _ in loader:
        xb = xb.view(xb.size(0), -1).to(device)
        optim.zero_grad()
        out, _ = model(xb)
        loss = criterion(out, xb)
        loss.backward()
        optim.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-perturb", type=int, default=9, help="number of latent perturbation steps")
    parser.add_argument("--perturb-scale", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = os.path.join("outputs", "part5")
    ensure_dir(outdir)

    transform = transforms.ToTensor()
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(testset,  batch_size=256, shuffle=False, num_workers=2)

    model = ManifoldAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} - loss {loss:.6f}")

    model.eval()
    xb, _ = next(iter(test_loader))
    img = xb[0:1].to(device)
    img_flat = img.view(1, -1)

    with torch.no_grad():
        out_clean, z = model(img_flat)
        z = z.squeeze(0).cpu().numpy()
    print("Latent (clean) vector:", z)

    rng = np.random.RandomState(0)
    seq = []
    with torch.no_grad():
        seq.append(img.cpu())
        seq.append(out_clean.cpu().view(1,1,28,28))

    for i in range(args.n_perturb):
        direction = rng.normal(size=z.shape)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        scale = args.perturb_scale * (i - args.n_perturb//2)
        z_pert = z + scale * direction
        z_pert_t = torch.tensor(z_pert, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model.decoder(z_pert_t).cpu().view(1,1,28,28)
        seq.append(recon)

    seq_cat = torch.cat(seq)
    utils.save_image(seq_cat, os.path.join(outdir, "latent_perturbations.png"), nrow=min(len(seq), 12), normalize=True)
    print("Saved latent perturbation reconstructions to", os.path.join(outdir, "latent_perturbations.png"))