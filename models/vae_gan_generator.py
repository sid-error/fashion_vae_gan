import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Generator, self).__init__()
        
        # Label embedding provides class blueprint
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        # Initial expansion of latent + class vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 64 * 7 * 7),
            nn.BatchNorm1d(64 * 7 * 7),
            nn.LeakyReLU(0.2) # Consistent with Encoder stability
        )
        
        # Smooth Upsampling (Resize-Convolution) to eliminate noise
        self.upsample_blocks = nn.Sequential(
            # Layer 1: 7x7 -> 14x14
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 2: 14x14 -> 28x28
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            # Final output mapped to [-1, 1]
            nn.Tanh()
        )

    def forward(self, z, labels):
        # 1. Map labels to class vectors
        c = self.label_emb(labels)
        
        # 2. Combine style (z) and class (c) information
        combined = torch.cat([z, c], dim=1)
        
        # 3. Project to feature map space
        x = self.fc(combined)
        x = x.view(-1, 64, 7, 7)
        
        # 4. Upsample to final fashion image
        img = self.upsample_blocks(x)
        return img