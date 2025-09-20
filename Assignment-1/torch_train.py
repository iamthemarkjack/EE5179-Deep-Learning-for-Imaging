import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tqdm import tqdm
import numpy as np

from torch_model import MLP

batch_size = 64
epochs = 15
lr = 1e-4
activations = ["relu", "sigmoid", "tanh"]
l2_values = [0.0, 1e-4, 1e-3, 1e-2]
results_dir = "torch_model_exps"
os.makedirs(results_dir, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_evaluate(act, l2_lambda):
    model = MLP(activation=act).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"{act} λ={l2_lambda} Epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Activation={act}, λ={l2_lambda}, Acc={acc:.4f}")
    plt.tight_layout()

    save_name = f"confmatrix_{act}_lambda{l2_lambda}".replace(".", "p")
    plt.savefig(os.path.join(results_dir, save_name))
    plt.close()

for act in activations:
    for l2 in l2_values:
        train_and_evaluate(act, l2)

print(f"All results saved in: {results_dir}")