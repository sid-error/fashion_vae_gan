import torch
import matplotlib.pyplot as plt
from models.vae_gan_wrapper import VAEGAN

def load_trained_model(checkpoint_path, latent_dim=128, num_classes=10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize with num_classes to match perfected architecture
    model = VAEGAN(latent_dim, num_classes).to(device)
    
    # Load the state dictionary from your RTX 3050 training run
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    return model

def generate_by_class(model, class_idx, num_samples=16, latent_dim=128, device=None):
    """
    Generates images specifically for one of the 10 fashion classes.
    class_idx: Integer from 0-9 (e.g., 8 for Bag, 3 for Dress)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        # 1. Sample random style vectors (z)
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # 2. Create a batch of the same label for the target class
        labels = torch.full((num_samples,), class_idx, dtype=torch.long).to(device)
        
        # 3. Generate conditioned images using the perfected Generator
        generated_images = model.generator(z, labels)

    return generated_images.cpu()

def plot_grid(images, class_name="Fashion Item"):
    # Visualization Engine for 4x4 grid
    plt.figure(figsize=(8, 8))
    for i in range(min(16, len(images))):
        plt.subplot(4, 4, i+1)
        # Denormalize: map [-1, 1] back to [0, 1] for matplotlib
        img = images[i][0] * 0.5 + 0.5
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Generated Samples: {class_name}")
    plt.show()

if __name__ == "__main__":
    # Example usage for testing your perfected model
    PATH = "checkpoints/vaegan_epoch_100.pth"
    CLASS_MAP = {0: "T-shirt", 3: "Dress", 8: "Bag", 9: "Ankle Boot"}
    
    target_class = 8 # Let's test Bags
    
    my_model = load_trained_model(PATH)
    samples = generate_by_class(my_model, class_idx=target_class)
    plot_grid(samples, class_name=CLASS_MAP[target_class])