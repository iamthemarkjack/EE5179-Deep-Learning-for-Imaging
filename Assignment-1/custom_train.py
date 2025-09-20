import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tqdm import tqdm

from custom_model import MLP

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

X_train = train_dataset.data.numpy().reshape(-1, 28*28) / 255.0
y_train = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy().reshape(-1, 28*28) / 255.0
y_test = test_dataset.targets.numpy()

epochs = 15
batch_size = 64
l2_values = [0.0, 1e-4, 1e-3, 1e-2]
activations = ["relu", "sigmoid", "tanh"]

results_dir = "custom_model_exps"
os.makedirs(results_dir, exist_ok=True)

def train_and_evaluate(act, l2_lambda):
    N = X_train.shape[0]
    net = MLP(input_dim=784,
              activations=(act, act, act),
              lr=0.01,
              l2_lambda=l2_lambda,
              seed=42)

    iteration = 0
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        pbar = tqdm(range(0, N, batch_size), desc=f"{act} λ={l2_lambda} epoch {epoch+1}/{epochs}", leave=False)
        for start in pbar:
            end = start + batch_size
            xb, yb = X_shuf[start:end], y_shuf[start:end]

            y_hat = net.forward(xb)
            loss = net.compute_loss(y_hat, yb)
            grads = net.backward(xb, yb)
            net.update_params(grads)
            iteration += 1

    y_pred = net.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    title = f"Activation={act}, λ={l2_lambda}, Acc={acc:.4f}"
    plt.title(title)
    plt.tight_layout()

    save_name = f"confmatrix_{act}_lambda{l2_lambda}".replace(".", "p")
    plt.savefig(os.path.join(results_dir, save_name))
    plt.close()

for act in activations:
    for l2 in l2_values:
        train_and_evaluate(act, l2)

print(f"All results saved in: {results_dir}")