import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_image_grid(tensor, path, nrow=8):
    utils.save_image(tensor, path, nrow=nrow, normalize=True)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 30)
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for (x, _) in dataloader:
        x = x.view(x.size(0), -1).to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, x)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate_pca_reconstruction(test_images_np, n_components=30):
    pca = PCA(n_components=n_components)
    pca.fit(test_images_np)
    projected = pca.transform(test_images_np)
    reconstructed = pca.inverse_transform(projected)
    return reconstructed, pca


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = os.path.join("outputs", "part1")
    ensure_dir(outdir)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(testset, batch_size=1000, shuffle=False)

    model = AE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs+1):
        loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} - Train MSE Loss: {loss:.6f}")

    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x_flat = x.view(x.size(0), -1).to(device)
        out = model(x_flat).cpu().view(-1, 1, 28, 28)

    compare = torch.cat([x[:64], out[:64]])
    save_image_grid(compare, os.path.join(outdir, "ae_reconstructions.png"), nrow=8)
    print("Saved AE reconstructions to", os.path.join(outdir, "ae_reconstructions.png"))

    total_mse = 0.0
    total_n = 0
    with torch.no_grad():
        for xb, _ in test_loader:
            xb_flat = xb.view(xb.size(0), -1).to(device)
            pred = model(xb_flat).cpu().numpy()
            xb_np = xb_flat.cpu().numpy()
            total_mse += ((pred - xb_np)**2).sum()
            total_n += xb_np.size
    ae_mse = total_mse / total_n
    print(f"Autoencoder test MSE: {ae_mse:.8f}")

    test_images = next(iter(test_loader))[0]
    test_images_np = test_images.view(test_images.size(0), -1).numpy()
    pca_recon, pca_model = evaluate_pca_reconstruction(test_images_np, n_components=30)
    pca_mse = ((pca_recon - test_images_np)**2).mean()
    print(f"PCA (30 comps) test MSE (on same test batch): {pca_mse:.8f}")

    pca_recon_torch = torch.tensor(pca_recon, dtype=torch.float).view(-1,1,28,28)
    compare2 = torch.cat([test_images[:64], pca_recon_torch[:64]])
    save_image_grid(compare2, os.path.join(outdir, "pca_reconstructions.png"), nrow=8)
    print("Saved PCA reconstructions to", os.path.join(outdir, "pca_reconstructions.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
