import torch
import torch.nn as nn
from .vae_encoder import VAEEncoder
from .vae_gan_generator import Generator
from .gan_discriminator import Discriminator

class VAEGAN(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(VAEGAN, self).__init__()
        # Module 2: Hybrid VAE-GAN Architecture
        self.encoder = VAEEncoder(latent_dim, num_classes)
        self.generator = Generator(latent_dim, num_classes) 
        self.discriminator = Discriminator(num_classes)

    def reparameterize(self, mu, logvar):
        # The Reparameterization Trick: z = mu + sigma * epsilon
        # This allows gradients to flow through the stochastic sampling process
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        # 1. Encoding Path: Map image and label to latent distribution
        mu, logvar = self.encoder(x, labels)
        
        # 2. Latent Sampling: Reparameterization for backpropagation
        z = self.reparameterize(mu, logvar)
        
        # 3. Decoding Path: Reconstruct image from style (z) and class label
        reconstructed_img = self.generator(z, labels)
        
        return reconstructed_img, mu, logvar