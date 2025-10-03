import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_mnist_dataloaders, plot_losses, show_predictions
import os, time

from model import CNN

def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model = CNN(use_batchnorm=args.use_batchnorm).to(device)

    train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size=args.batch_size, data_dir=args.data_dir)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float('inf')

    start_time = time.time()
    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_running, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_running += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_loss = val_running / len(val_loader.dataset)
        val_acc = correct / total
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, "mnist_cnn_best.pt"))

    elapsed = time.time() - start_time

    model.load_state_dict(torch.load(os.path.join(args.save_dir, "mnist_cnn_best.pt"), map_location=device))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    test_acc = correct / total
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Training time: {elapsed:.2f} seconds")

    plot_losses(history, os.path.join(args.save_dir, "losses.png"))
    show_predictions(model, test_loader, device=device, n=16, save_path=os.path.join(args.save_dir, "preds.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--use_batchnorm", action="store_true")
    args = parser.parse_args()
    train(args)