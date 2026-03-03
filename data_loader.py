import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_fashion_mnist_loaders(batch_size=128, filter_classes=None):
    # Standardizing the transformation for 28x28 grayscale
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Maps [0, 1] to [-1, 1]
    ])

    # Download and load training and test sets
    train_set = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_set = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Logic to filter for specific experiments or use all 10 classes
    if filter_classes:
        train_indices = [i for i, (_, label) in enumerate(train_set) if label in filter_classes]
        train_dataset = Subset(train_set, train_indices)
        
        test_indices = [i for i, (_, label) in enumerate(test_set) if label in filter_classes]
        test_dataset = Subset(test_set, test_indices)
    else:
        train_dataset = train_set
        test_dataset = test_set
    
    # Loaders for the RTX 3050 (Module 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Test with all 10 classes for the perfect model
    train_l, test_l = get_fashion_mnist_loaders(filter_classes=None)
    print(f"Dataset Prepared: {len(train_l.dataset)} training images, {len(test_l.dataset)} test images.")