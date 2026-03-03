import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(VAEEncoder, self).__init__()
        
        # Label embedding creates a 64-dimensional class vector
        self.label_embedding = nn.Embedding(num_classes, 64)
        
        # Convolutional backbone for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2), # Better than ReLU for GAN stability
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        
        # Combined size: (64*7*7 image features) + (64 label features)
        combined_dim = (64 * 7 * 7) + 64
        
        # Predict class-conditioned distribution parameters
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_dim, latent_dim)

    def forward(self, x, labels):
        # 1. Extract visual features from the fashion image
        x = self.feature_extractor(x)
        
        # 2. Get the specific vector for the chosen class
        label_features = self.label_embedding(labels)
        
        # 3. Concatenate image data with class label context
        combined = torch.cat([x, label_features], dim=1)
        
        # 4. Return the latent distribution (mu and log-variance)
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar