import torch
import numpy as np
from sklearn.metrics import accuracy_score 

def calculate_metrics(model, test_loader, device):
    model.eval()
    recon_losses = []
    
    # Storage for binary classification results (Real vs Fake)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 1. Structural Coherence (Conditioned Reconstruction) 
            # We pass labels so the model knows which class shape to reconstruct
            reconstructed, _, _ = model(images, labels)
            recon_loss = torch.mean(torch.abs(images - reconstructed))
            recon_losses.append(recon_loss.item())

            # 2. Visual Realism (Discriminator Accuracy)
            # Test how often the discriminator identifies real images + correct labels as 1
            preds_real = model.discriminator(images, labels).cpu().numpy()
            all_preds.extend(preds_real > 0.5)
            all_targets.extend(np.ones(len(preds_real)))
            
            # 3. Generative Quality
            # Sample random noise and use the ground-truth labels for generation
            z = torch.randn(len(images), 128).to(device)
            fakes = model.generator(z, labels)
            
            # Test how often it identifies generated images + class labels as 0
            preds_fake = model.discriminator(fakes, labels).cpu().numpy()
            all_preds.extend(preds_fake > 0.5)
            all_targets.extend(np.zeros(len(preds_fake)))

    # Calculate average metrics
    avg_recon = np.mean(recon_losses)
    disc_acc = accuracy_score(all_targets, all_preds)
    
    return {
        "Mean Absolute Error": avg_recon, # Low value = better shape preservation
        "Discriminator Accuracy": disc_acc # Goal is 0.5 (Perfect Equilibrium)
    }