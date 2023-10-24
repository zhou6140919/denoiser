import torch
import torch.nn as nn


class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()

        # Encoder
        # Single channel (grayscale), 64 filters, 3x3 kernels
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        # Decoder
        self.dec1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Output single channel (grayscale)
        self.dec3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Activation function
        self.act = nn.ReLU()

    def forward(self, x):
        # Encoder
        x = self.act(self.enc1(x))
        x = self.act(self.enc2(x))
        x = self.act(self.enc3(x))

        # Decoder
        x = self.act(self.dec1(x))
        x = self.act(self.dec2(x))
        # Sigmoid activation to ensure output is between 0 and 1
        x = torch.sigmoid(self.dec3(x))

        return x


class DnCNN(nn.Module):
    def __init__(self, channels=3, num_layers=17):  # Default channels set to 3 for RGB
        super(DnCNN, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Conv2d(channels, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))

        # Output layer
        # Output channels same as input
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
