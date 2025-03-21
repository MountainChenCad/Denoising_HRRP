# cae_models.py (corrected version)
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoEncoder(nn.Module):
    """
    Convolutional AutoEncoder for HRRP signal denoising.
    Uses 1D convolutional layers for encoding and transposed convolutional layers for decoding.
    """

    def __init__(self, input_dim=500, latent_dim=64, hidden_dim=128):
        """
        Parameters:
            input_dim (int): Dimension of input HRRP sequence
            latent_dim (int): Dimension of latent space representation
            hidden_dim (int): Dimension of hidden layers
        """
        super(ConvAutoEncoder, self).__init__()

        # Store dimensions
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(True)
        )

        # Calculate the output size after encoder
        # For each conv layer with stride 2, the size is ceil(L/2)
        # We apply this 3 times for 3 conv layers
        self.encoded_dim = input_dim
        for _ in range(3):  # 3 layers with stride 2
            self.encoded_dim = (self.encoded_dim + 1) // 2  # ceiling division

        # Calculate flattened features size
        self.flattened_size = hidden_dim * 4 * self.encoded_dim

        # Bottleneck fully connected layers
        self.fc_encode = nn.Linear(self.flattened_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.ConvTranspose1d(hidden_dim, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

        print(f"Encoder output dim: {self.encoded_dim}")
        print(f"Flattened size: {self.flattened_size}")

    def encode(self, x):
        """
        Encode HRRP data to latent representation

        Parameters:
            x (torch.Tensor): Input HRRP data [batch_size, input_dim]

        Returns:
            torch.Tensor: Latent representation [batch_size, latent_dim]
        """
        # Reshape for 1D convolution [batch_size, channels, length]
        x = x.unsqueeze(1)  # Add channel dimension

        # Apply encoder layers
        x = self.encoder(x)

        # Flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Ensure x has the expected shape
        if x.size(1) != self.flattened_size:
            print(f"Warning: encoder output size {x.size(1)} doesn't match expected {self.flattened_size}")
            # We can pad or truncate to match expected size if needed
            if x.size(1) > self.flattened_size:
                x = x[:, :self.flattened_size]
            else:
                padding = torch.zeros(batch_size, self.flattened_size - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)

        # Reduce to latent space
        latent = self.fc_encode(x)

        return latent

    def decode(self, latent):
        """
        Decode from latent representation to reconstructed HRRP

        Parameters:
            latent (torch.Tensor): Latent representation [batch_size, latent_dim]

        Returns:
            torch.Tensor: Reconstructed HRRP data [batch_size, input_dim]
        """
        # Expand from latent space
        x = self.fc_decode(latent)

        # Reshape for transposed convolution
        batch_size = x.size(0)
        x = x.view(batch_size, self.hidden_dim * 4, self.encoded_dim)

        # Apply decoder layers
        x = self.decoder(x)

        # Reshape to original format [batch_size, input_dim]
        x = x.squeeze(1)

        # Ensure output has correct size
        if x.size(1) != self.input_dim:
            # Resize using interpolation to match original dimension
            x = F.interpolate(x.unsqueeze(1), size=self.input_dim, mode='linear').squeeze(1)

        return x

    def forward(self, x):
        """
        Forward pass through autoencoder

        Parameters:
            x (torch.Tensor): Input HRRP data [batch_size, input_dim]

        Returns:
            torch.Tensor: Reconstructed HRRP data [batch_size, input_dim]
            torch.Tensor: Latent representation [batch_size, latent_dim]
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent