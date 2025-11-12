import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

class BinaryAddDataset(Dataset):
    def __init__(self, n_samples=10000, seq_len=8):
        self.data = []
        for _ in range(n_samples):
            a = random.randint(0, 2**seq_len - 1)
            b = random.randint(0, 2**seq_len - 1)
            s = a + b
            a_bits = [(a >> i) & 1 for i in range(seq_len)]
            b_bits = [(b >> i) & 1 for i in range(seq_len)]
            sum_bits = [(s >> i) & 1 for i in range(seq_len + 1)]
            x = torch.tensor(list(zip(a_bits, b_bits)), dtype=torch.float32)
            y = torch.tensor(sum_bits, dtype=torch.float32)
            self.data.append((x, y))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class AddLSTM(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out).squeeze(-1)  # (batch, seq_len)
        return logits

def train_model(seq_len, hidden_size, loss_type, device, epochs=10):
    train_set = BinaryAddDataset(n_samples=20000, seq_len=seq_len)
    test_set = BinaryAddDataset(n_samples=2000, seq_len=seq_len)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128)

    model = AddLSTM(hidden_size=hidden_size).to(device)
    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses, accs = [], []
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total_bits = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            preds = torch.sigmoid(logits)
            target = y[:, :x.size(1)]
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (preds.round() == target).sum().item()
            total_bits += preds.numel()
        losses.append(running_loss / len(train_loader.dataset))
        accs.append(correct / total_bits)
    return model, losses, accs

def test_on_lengths(model, device, loss_type, hidden_size, lengths):
    results = []
    model.eval()
    with torch.no_grad():
        for L in lengths:
            test_set = BinaryAddDataset(n_samples=1000, seq_len=L)
            loader = DataLoader(test_set, batch_size=128)
            correct, total = 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = torch.sigmoid(logits) > 0.5
                target = y[:, :x.size(1)].bool()
                correct += (preds == target).sum().item()
                total += preds.numel()
            acc = correct / total
            results.append((hidden_size, loss_type, L, acc))
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs_binary_experiments", exist_ok=True)

    train_lengths = [3, 5, 10]
    test_lengths = list(range(1, 21))
    hidden_sizes = [8, 16, 32]
    losses = ["bce", "mse"]

    summary_records = []

    for loss_type in losses:
        plt.figure(figsize=(12, 5))
        sns.set(style="whitegrid")

        for hidden_size in hidden_sizes:
            for L_train in train_lengths:
                print(f"Training LSTM: L_train={L_train}, hidden={hidden_size}, loss={loss_type}")
                model, losses_curve, acc_curve = train_model(
                    seq_len=L_train, hidden_size=hidden_size, loss_type=loss_type, device=device, epochs=8
                )
                plt.plot(acc_curve, label=f"L={L_train}, h={hidden_size}")
                results = test_on_lengths(model, device, loss_type, hidden_size, test_lengths)
                summary_records.extend(results)

        plt.title(f"Training Accuracy Curves ({loss_type.upper()})")
        plt.xlabel("Epoch")
        plt.ylabel("Bit Accuracy")
        plt.legend()
        plt.savefig(f"outputs_binary_experiments/train_curves_{loss_type}.png")
        plt.close()

    df = pd.DataFrame(summary_records, columns=["hidden_size", "loss", "L_test", "bit_accuracy"])
    df.to_csv("outputs_binary_experiments/summary_results.csv", index=False)

    for loss_type in losses:
        plt.figure(figsize=(10, 6))
        subset = df[df["loss"] == loss_type]
        sns.lineplot(data=subset, x="L_test", y="bit_accuracy", hue="hidden_size", marker="o")
        plt.title(f"Bit Accuracy vs Sequence Length ({loss_type.upper()})")
        plt.xlabel("Sequence Length (L)")
        plt.ylabel("Bit Accuracy")
        plt.legend(title="Hidden Size")
        plt.savefig(f"outputs_binary_experiments/bit_accuracy_vs_len_{loss_type}.png")
        plt.close()

    for loss_type in losses:
        plt.figure(figsize=(10, 6))
        subset = df[df["loss"] == loss_type]
        pivot = subset.groupby(["hidden_size", "L_test"], as_index=False)["bit_accuracy"].mean()
        pivot = pivot.pivot(index="hidden_size", columns="L_test", values="bit_accuracy")
        sns.heatmap(pivot, cmap="coolwarm", annot=False)
        plt.title(f"Heatmap: Bit Accuracy (Hidden Size vs Length) [{loss_type.upper()}]")
        plt.xlabel("Sequence Length (L)")
        plt.ylabel("Hidden Size")
        plt.savefig(f"outputs_binary_experiments/heatmap_{loss_type}.png")
        plt.close()

    print("\n All experiments completed. Results saved in outputs_binary_experiments/")
    print(df.groupby(["loss", "hidden_size"])["bit_accuracy"].mean())

if __name__ == "__main__":
    main()