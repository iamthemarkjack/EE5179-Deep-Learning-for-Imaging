import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from model import CNN

def load_model(model_path, use_batchnorm=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(use_batchnorm=use_batchnorm)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def non_targeted_attack(model, device, target_class, num_iterations=1000, 
                       step_size=1.0, save_path='non_targeted_attack.png'):
    print(f"\nGenerating adversarial example for target class: {target_class}")
    
    X = torch.randn(1, 1, 28, 28, device=device) * 0.3 + 0.5
    X = X.clamp(0, 1)
    X.requires_grad = True
    
    cost_history = []
    
    optimizer = torch.optim.SGD([X], lr=step_size)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        output = model(X)
        
        cost = output[0, target_class]
        
        cost.backward()
        
        with torch.no_grad():
            X.data = X.data + step_size * X.grad.data
            X.data = X.data.clamp(0, 1)
        
        cost_history.append(cost.item())
        
        if (iteration + 1) % 200 == 0:
            with torch.no_grad():
                probs = F.softmax(output, dim=1)
                pred_class = output.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
            print(f"Iteration {iteration + 1}: Cost = {cost.item():.4f}, "
                  f"Predicted = {pred_class}, Confidence = {confidence:.4f}")
    
    with torch.no_grad():
        final_output = model(X)
        probs = F.softmax(final_output, dim=1)
        pred_class = final_output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    return X.detach(), cost_history, pred_class, confidence

def targeted_attack(model, device, target_image, target_class, num_iterations=1000,
                   step_size=1.0, beta=0.0001, save_path='targeted_attack.png'):
    print(f"\nGenerating adversarial example that looks like digit "
          f"{target_image} but classified as {target_class}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                   download=True, transform=transform)
    
    target_img_tensor = None
    for img, label in train_dataset:
        if label == target_image:
            target_img_tensor = img.unsqueeze(0).to(device)
            break
    
    if target_img_tensor is None:
        raise ValueError(f"Could not find image with label {target_image}")
    
    X = torch.randn(1, 1, 28, 28, device=device) * 0.3 + 0.5
    X = X.clamp(0, 1)
    X.requires_grad = True
    
    cost_history = []
    
    optimizer = torch.optim.SGD([X], lr=step_size)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        output = model(X)
        
        mse_loss = F.mse_loss(X, target_img_tensor)
        cost = output[0, target_class] - beta * mse_loss
        
        cost.backward()
        
        with torch.no_grad():
            X.data = X.data + step_size * X.grad.data
            X.data = X.data.clamp(0, 1)
        
        cost_history.append(cost.item())
        
        if (iteration + 1) % 200 == 0:
            with torch.no_grad():
                probs = F.softmax(output, dim=1)
                pred_class = output.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
            print(f"Iteration {iteration + 1}: Cost = {cost.item():.4f}, "
                  f"Predicted = {pred_class}, Confidence = {confidence:.4f}, "
                  f"MSE = {mse_loss.item():.4f}")
    
    with torch.no_grad():
        final_output = model(X)
        probs = F.softmax(final_output, dim=1)
        pred_class = final_output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    return X.detach(), target_img_tensor, cost_history, pred_class, confidence

def run_non_targeted_experiments(model, device, save_dir):
    print("\n" + "="*70)
    print("NON-TARGETED ATTACK")
    print("="*70)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Non-Targeted Adversarial Examples (Generated from Noise)', fontsize=14)
    
    all_cost_histories = []
    results = []
    
    for target_class in range(10):
        adv_img, cost_history, pred_class, confidence = non_targeted_attack(
            model, device, target_class, num_iterations=1000, step_size=1.0
        )
        
        all_cost_histories.append(cost_history)
        results.append((target_class, pred_class, confidence))
        
        ax = axes[target_class // 5, target_class % 5]
        img_np = adv_img.squeeze().cpu().numpy()
        ax.imshow(img_np, cmap='gray')
        ax.set_title(f'Target: {target_class}\nPred: {pred_class} ({confidence:.2f})', 
                    fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    save_file_path = os.path.join(save_dir, 'non_targeted_all_classes.png')
    plt.savefig(save_file_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i, cost_history in enumerate(all_cost_histories):
        ax.plot(cost_history, label=f'Class {i}', alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost (Logit Value)', fontsize=12)
    ax.set_title('Cost Function over Iterations - Non-Targeted Attack', fontsize=14)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_file_path = os.path.join(save_dir, 'non_targeted_cost.png')
    plt.savefig(save_file_path, dpi=150, bbox_inches='tight')
    plt.show()
    correct = sum(1 for t, p, _ in results if t == p)
    print(f"\nCorrect predictions: {correct}/10")

def run_targeted_experiments(model, device, save_dir):
    print("\n" + "="*70)
    print("TARGETED ATTACK")
    print("="*70)

    digit_pairs = [
        (0, 6), (1, 7), (2, 5), (3, 8), (4, 9),
        (5, 3), (6, 8), (7, 1), (8, 3), (9, 4)
    ]
    
    fig, axes = plt.subplots(2, 10, figsize=(20, 6))
    fig.suptitle('Targeted Adversarial Examples', fontsize=16)
    
    all_cost_histories = []
    results = []
    
    for idx, (source_digit, target_class) in enumerate(digit_pairs):
        adv_img, target_img, cost_history, pred_class, confidence = targeted_attack(
            model, device, source_digit, target_class, 
            num_iterations=1000, step_size=1.0, beta=100
        )
        
        all_cost_histories.append(cost_history)
        results.append((source_digit, target_class, pred_class, confidence))
        
        ax = axes[0, idx]
        ax.imshow(target_img.squeeze().cpu().numpy(), cmap='gray')
        if idx == 0:
            ax.set_ylabel('Original', fontsize=10)
        ax.set_title(f'Digit {source_digit}', fontsize=9)
        
        ax = axes[1, idx]
        ax.imshow(adv_img.squeeze().cpu().numpy(), cmap='gray')
        if idx == 0:
            ax.set_ylabel('Generated', fontsize=10)
        ax.set_title(f'Target: {target_class}', fontsize=9)

    plt.tight_layout()
    save_file_path = os.path.join(save_dir, 'targeted_all_pairs.png')
    plt.savefig(save_file_path, dpi=150)
    plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i, ((source, target), cost_history) in enumerate(zip(digit_pairs, all_cost_histories)):
        ax.plot(cost_history, label=f'{source}→{target}', alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost (Logit - beta*MSE)', fontsize=12)
    ax.set_title('Cost Function over Iterations - Targeted Attack', fontsize=14)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_file_path = os.path.join(save_dir, 'targeted_cost.png')
    plt.savefig(save_file_path, dpi=150, bbox_inches='tight')
    plt.show()
    correct = sum(1 for _, t, p, _ in results if t == p)
    print(f"\nSuccessful attacks: {correct}/10")

def main(args):    
    model_path = args.model_path
    batch_size = args.batch_size
    use_batchnorm = args.use_batchnorm
    save_dir = args.save_dir

    print("Loading model...")
    model, device = load_model(model_path, use_batchnorm=use_batchnorm)
    print(f"Model loaded successfully on {device}")
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    print("\n" + "="*70)
    print("ADVERSARIAL EXAMPLES GENERATION")
    print("="*70)
    
    run_non_targeted_experiments(model, device, save_dir)
    
    run_targeted_experiments(model, device, save_dir)
    
    print("\n" + "="*70)
    print("All adversarial experiments completed!")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/mnist_cnn_best.pt',)
    parser.add_argument('--use_batchnorm', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./images')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    main(args)