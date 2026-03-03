import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import csv
from models.vae_gan_wrapper import VAEGAN
from data_loader import get_fashion_mnist_loaders
from loss_functions import compute_vae_gan_loss

# 1. Configuration & Model Initialization
latent_dim = 128
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAEGAN(latent_dim, num_classes).to(device)

# 2. Optimizers (TTUR: Generator learns faster than Discriminator)
opt_vae = optim.Adam(
    list(model.encoder.parameters()) + list(model.generator.parameters()), 
    lr=0.0002, 
    betas=(0.5, 0.999)
)
opt_dist = optim.Adam(
    model.discriminator.parameters(), 
    lr=0.00005, # Slower learning rate for discriminator stability
    betas=(0.5, 0.999)
)

def log_metrics(epoch, recon, kl, adv, feat, total):
    log_file = 'experiment_results.csv'
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Recon_Loss', 'KL_Loss', 'Adv_Loss', 'Feat_Loss', 'Total_Loss'])
        writer.writerow([epoch, recon, kl, adv, feat, total])

def save_checkpoint(epoch, model, opt_vae, opt_dist):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    checkpoint_path = os.path.join('checkpoints', f'vaegan_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    print(f"--- Checkpoint saved: {checkpoint_path} ---")

def train_one_epoch(epoch, loader):
    model.train()
    metrics = {'recon': 0, 'kl': 0, 'adv': 0, 'feat': 0, 'total': 0}
    
    for batch_idx, (real_imgs, labels) in enumerate(loader):
        real_imgs, labels = real_imgs.to(device), labels.to(device)
        
        # --- Train VAE & Generator ---
        opt_vae.zero_grad()
        
        # Pass labels for conditional encoding/decoding
        recon_imgs, mu, logvar = model(real_imgs, labels)
        
        # Get Discriminator outputs for Feature Matching and Adversarial Loss
        # We need the internal features (feat) and the final prediction (logit)
        # Note: Your updated Discriminator forward must return both
        real_logit = model.discriminator(real_imgs, labels)
        fake_logit = model.discriminator(recon_imgs, labels)
        
        # For simplicity in this step, we use the output of conv_layers as features
        # In a refined setup, the discriminator forward returns (logit, features)
        real_feat = model.discriminator.conv_layers(real_imgs)
        fake_feat = model.discriminator.conv_layers(recon_imgs)

        loss, recon, kl, adv, feat = compute_vae_gan_loss(
            recon_imgs, real_imgs, mu, logvar, real_feat, fake_feat, fake_logit
        )
        
        loss.backward()
        opt_vae.step()

        # --- Train Discriminator ---
        opt_dist.zero_grad()
        
        # Discriminator learns to separate real class items from fake class items
        d_real = model.discriminator(real_imgs, labels)
        d_fake = model.discriminator(recon_imgs.detach(), labels)
        
        loss_d = (F.binary_cross_entropy(d_real, torch.ones_like(d_real)) + 
                  F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))) / 2
        
        loss_d.backward()
        opt_dist.step()

        # Update running metrics
        metrics['recon'] += recon.item(); metrics['kl'] += kl.item()
        metrics['adv'] += adv.item(); metrics['feat'] += feat.item()
        metrics['total'] += loss.item()

    # Average metrics for the epoch
    avg = {k: v / len(loader) for k, v in metrics.items()}
    print(f"Epoch {epoch} | Total Loss: {avg['total']:.4f} | Recon: {avg['recon']:.4f}")
    return avg

if __name__ == "__main__":
    # Load all 10 classes for the perfected model
    train_loader, _ = get_fashion_mnist_loaders(batch_size=128, filter_classes=None)
    
    for epoch in range(1, 101):
        m = train_one_epoch(epoch, train_loader)
        log_metrics(epoch, m['recon'], m['kl'], m['adv'], m['feat'], m['total'])
        
        if epoch % 10 == 0:
            save_checkpoint(epoch, model, opt_vae, opt_dist)