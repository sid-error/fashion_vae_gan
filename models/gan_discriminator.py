import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        
        # New: Label Embedding to provide class context to the discriminator
        self.label_emb = nn.Embedding(num_classes, 64 * 7 * 7)
        
        # Initial layers to process the image
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # 14x14
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Flatten()
        )
        
        # Final classification layer (Combined image features + label features)
        # We use 64 * 7 * 7 * 2 because we concatenate the two vectors
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 * 2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # 1. Extract features from the image
        img_features = self.conv_layers(x)
        
        # 2. Get features for the provided class label
        label_features = self.label_emb(labels)
        
        # 3. Concatenate image and label features
        combined = torch.cat([img_features, label_features], dim=1)
        
        # 4. Determine probability of being real AND matching the class
        validity = self.fc(combined)
        return validity