# ae_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """
    Fully connected layer-based AutoEncoder for HRRP signal denoising.
    Uses linear layers for encoding and decoding, simpler than CAE but still effective.
    """

    def __init__(self, input_dim=500, latent_dim=64, hidden_dim=256):
        """
        Parameters:
            input_dim (int): Dimension of input HRRP sequence
            latent_dim (int): Dimension of latent space representation
            hidden_dim (int): Dimension of hidden layers
        """
        super(AutoEncoder, self).__init__()

        # Store dimensions
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(True)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output range [0, 1]
        )

    def encode(self, x):
        """
        Encode HRRP data to latent representation

        Parameters:
            x (torch.Tensor): Input HRRP data [batch_size, input_dim]

        Returns:
            torch.Tensor: Latent representation [batch_size, latent_dim]
        """
        return self.encoder(x)

    def decode(self, latent):
        """
        Decode from latent representation to reconstructed HRRP

        Parameters:
            latent (torch.Tensor): Latent representation [batch_size, latent_dim]

        Returns:
            torch.Tensor: Reconstructed HRRP data [batch_size, input_dim]
        """
        return self.decoder(latent)

    def forward(self, x):
        """
        Forward pass of autoencoder

        Parameters:
            x (torch.Tensor): Input HRRP data [batch_size, input_dim]

        Returns:
            torch.Tensor: Reconstructed HRRP data [batch_size, input_dim]
            torch.Tensor: Latent representation [batch_size, latent_dim]
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent