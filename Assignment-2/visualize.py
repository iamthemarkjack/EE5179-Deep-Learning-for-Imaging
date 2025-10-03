import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from model import CNN

def load_model(model_path, use_batchnorm=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(use_batchnorm=use_batchnorm)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def plot_conv_filters(model, layer_name='conv1', save_path='conv1_filters.png'):
    if layer_name == 'conv1':
        weights = model.conv1.weight.data.cpu()
    elif layer_name == 'conv2':
        weights = model.conv2.weight.data.cpu()
    else:
        raise ValueError("Layer name should be 'conv1' or 'conv2'")
    
    num_filters = weights.shape[0]
    
    if layer_name == 'conv1':
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        fig.suptitle(f'{layer_name.upper()} Filters (3x3)', fontsize=16)
        
        for idx in range(num_filters):
            ax = axes[idx // 8, idx % 8]
            filter_img = weights[idx, 0, :, :].numpy()
            im = ax.imshow(filter_img, cmap='gray')
            ax.axis('off')
            ax.set_title(f'F{idx}', fontsize=8)
        
        plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Conv1 filters saved to {save_path}")
        
    else:
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        fig.suptitle(f'{layer_name.upper()} Filters (First Input Channel, 3x3)', fontsize=16)
        
        for idx in range(min(32, num_filters)):
            ax = axes[idx // 8, idx % 8]
            filter_img = weights[idx, 0, :, :].numpy()
            im = ax.imshow(filter_img, cmap='gray')
            ax.axis('off')
            ax.set_title(f'F{idx}', fontsize=8)
        
        plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Conv2 filters saved to {save_path}")

def visualize_activations(model, device, test_loader, num_images=3, 
                         save_path='activations.png'):
    model.eval()
    
    images, labels = next(iter(test_loader))
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        activations = model.get_activations(images)
    
    for img_idx in range(num_images):
        fig = plt.figure(figsize=(20, 10))
        
        ax = plt.subplot(3, 9, 1)
        ax.imshow(images[img_idx, 0].cpu().numpy(), cmap='gray')
        ax.set_title('Original', fontsize=10)
        ax.axis('off')
        
        conv1_act = activations['conv1'][img_idx].cpu().numpy()
        for i in range(8):
            ax = plt.subplot(3, 9, i + 2)
            ax.imshow(conv1_act[i], cmap='viridis')
            ax.set_title(f'Conv1-{i}', fontsize=8)
            ax.axis('off')
        
        pool1_act = activations['pool1'][img_idx].cpu().numpy()
        for i in range(8):
            ax = plt.subplot(3, 9, i + 10)
            ax.imshow(pool1_act[i], cmap='viridis')
            ax.set_title(f'Pool1-{i}', fontsize=8)
            ax.axis('off')
        
        conv2_act = activations['conv2'][img_idx].cpu().numpy()
        for i in range(8):
            ax = plt.subplot(3, 9, i + 19)
            ax.imshow(conv2_act[i], cmap='viridis')
            ax.set_title(f'Conv2-{i}', fontsize=8)
            ax.axis('off')
        
        plt.suptitle(f'Feature Map Activations - Image {img_idx + 1} (Label: {labels[img_idx].item()})', 
                     fontsize=14)
        plt.tight_layout()
        save_name = save_path.replace('.png', f'_img{img_idx}.png')
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Activations for image {img_idx} saved to {save_name}")

def occlusion_experiment(model, device, test_loader, occluder_size=3, 
                        num_images=5, save_path='occlusion.png'):
    model.eval()
    
    images, labels = next(iter(test_loader))
    
    correct_images = []
    correct_labels = []
    
    with torch.no_grad():
        for img, label in zip(images, labels):
            if len(correct_images) >= num_images:
                break
            img_tensor = img.unsqueeze(0).to(device)
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            if pred == label.item():
                correct_images.append(img)
                correct_labels.append(label.item())
    
    for img_idx, (img, label) in enumerate(zip(correct_images, correct_labels)):
        img_np = img.squeeze().numpy()
        H, W = img_np.shape
        
        prob_map = np.zeros((H - occluder_size + 1, W - occluder_size + 1))
        
        for i in range(0, H - occluder_size + 1):
            for j in range(0, W - occluder_size + 1):
                occluded_img = img_np.copy()
                occluded_img[i:i+occluder_size, j:j+occluder_size] = 0.5
                
                occluded_tensor = torch.FloatTensor(occluded_img).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(occluded_tensor)
                    prob = F.softmax(output, dim=1)[0, label].item()
                
                prob_map[i, j] = prob
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title(f'Original Image\nTrue Label: {label}', fontsize=12)
        axes[0].axis('off')
        
        im = axes[1].imshow(prob_map, cmap='hot', interpolation='nearest')
        axes[1].set_title(f'Probability Map for Class {label}\n(Occluder Size: {occluder_size}x{occluder_size})', 
                         fontsize=12)
        axes[1].set_xlabel('X Position')
        axes[1].set_ylabel('Y Position')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[2].imshow(img_np, cmap='gray', alpha=0.5)
        axes[2].imshow(prob_map, cmap='hot', alpha=0.5, interpolation='nearest')
        axes[2].set_title('Overlay', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        save_name = save_path.replace('.png', f'_img{img_idx}.png')
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Occlusion experiment for image {img_idx} saved to {save_name}")


def main(args):
    model_path = args.model_path
    batch_size = args.batch_size
    use_batchnorm = args.use_batchnorm
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, 
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    print("Loading model...")
    model, device = load_model(model_path, use_batchnorm=use_batchnorm)
    print(f"Model loaded successfully on {device}")
    
    print("\n" + "="*50)
    print("Plotting Conv1 Filters")
    print("="*50)
    save_file_path = os.path.join(save_dir, 'conv1_filters.png')
    plot_conv_filters(model, layer_name='conv1', save_path=save_file_path)
    
    print("\n" + "="*50)
    print("Plotting Conv2 Filters")
    print("="*50)
    save_file_path = os.path.join(save_dir, 'conv2_filters.png')
    plot_conv_filters(model, layer_name='conv2', save_path=save_file_path)
    
    print("\n" + "="*50)
    print("Visualizing Activations")
    print("="*50)
    save_file_path = os.path.join(save_dir, 'activations.png')
    visualize_activations(model, device, test_loader, num_images=3, save_path=save_file_path)
    
    print("\n" + "="*50)
    print("Occlusion Experiment")
    print("="*50)
    save_file_path = os.path.join(save_dir, 'occlusion.png')
    occlusion_experiment(model, device, test_loader, occluder_size=3, num_images=5, save_path=save_file_path)
    
    print("\n" + "="*50)
    print("All visualization tasks completed!")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/mnist_cnn_best.pt',)
    parser.add_argument('--use_batchnorm', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./images')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    main(args)