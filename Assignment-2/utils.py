import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os

def get_mnist_dataloaders(batch_size=128, data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainval = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_set, val_set = random_split(trainval, [50000, 10000])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def plot_losses(history, save_path=None):
    plt.figure(figsize=(8,5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def show_predictions(model, loader, device="cpu", n=16, save_path=None):
    model.eval()
    imgs, labs, preds = [], [], []
    with torch.no_grad():
        for x,y in loader:
            imgs.append(x)
            labs.append(y)
            logits = model(x.to(device))
            preds.append(torch.argmax(logits, dim=1).cpu())
            if len(imgs)*x.size(0) >= n:
                break
    imgs = torch.cat(imgs, dim=0)[:n]
    labs = torch.cat(labs, dim=0)[:n]
    preds = torch.cat(preds, dim=0)[:n]
    imgs = imgs * 0.3081 + 0.1307
    imgs = imgs.numpy()
    plt.figure(figsize=(8,8))
    for i in range(n):
        plt.subplot(int(np.sqrt(n)), int(np.sqrt(n)), i+1)
        plt.imshow(imgs[i,0], cmap="gray")
        plt.title(f"T:{int(labs[i])} P:{int(preds[i])}")
        plt.axis("off")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()