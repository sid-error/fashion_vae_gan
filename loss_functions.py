import torch
import torch.nn.functional as F

def compute_vae_gan_loss(reconstructed_x, x, mu, logvar, disc_real_feat, disc_fake_feat, disc_fake_logit):
    """
    disc_real_feat: Internal features of real images from Discriminator
    disc_fake_feat: Internal features of reconstructed images from Discriminator
    disc_fake_logit: Final probability output (Sigmoid) for fake images
    """
    
    # 1. Structural Reconstruction Loss (Pixel Level)
    # Using L1 (MAE) is better for preventing blurriness than MSE
    recon_loss = F.l1_loss(reconstructed_x, x)
    
    # 2. Feature Matching Loss (Deep Shape Level)
    # This forces the generator to produce the 'features' of the 10 classes
    feat_loss = F.mse_loss(disc_fake_feat, disc_real_feat)
    
    # 3. KL Divergence (Latent Regularization)
    # Standard VAE loss to keep the latent space continuous
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 4. Adversarial Loss (Realism Level)
    # The Generator wants the Discriminator to think these are real (Target = 1)
    adv_loss = F.binary_cross_entropy(disc_fake_logit, torch.ones_like(disc_fake_logit))
    
    # Balanced weighting for 10-class recognition
    # We prioritize Feature Matching and Recon to establish shapes first
    total_loss = (1 * kl_loss) + (50 * recon_loss) + (10 * feat_loss) + (1 * adv_loss)
    
    return total_loss, recon_loss, kl_loss, adv_loss, feat_loss