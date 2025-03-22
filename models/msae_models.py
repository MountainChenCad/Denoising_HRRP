# msae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ModifiedSparseAutoEncoder(nn.Module):
    """
    Modified Sparse AutoEncoder for HRRP signal denoising.
    Combines traditional autoencoder architecture with sparsity constraints and
    weight modification techniques to improve denoising performance.
    """

    def __init__(self, input_dim=500, latent_dim=64, hidden_dim=256, sparsity_param=0.05,
                 reg_lambda=0.0001, sparsity_beta=3):
        """
        Parameters:
            input_dim (int): Dimension of input HRRP sequence
            latent_dim (int): Dimension of latent space representation
            hidden_dim (int): Dimension of hidden layers
            sparsity_param (float): Target sparsity parameter (p), determines desired sparsity level
            reg_lambda (float): Weight regularization parameter (λ)
            sparsity_beta (float): Sparsity weight parameter (β)
        """
        super(ModifiedSparseAutoEncoder, self).__init__()

        # Store dimensions and regularization parameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.sparsity_param = sparsity_param
        self.reg_lambda = reg_lambda
        self.sparsity_beta = sparsity_beta

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Sigmoid()  # Using sigmoid to enforce activations between 0 and 1 for sparsity
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output range [0, 1]
        )

        # Initialize the weights with a modified approach
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights using a modified approach to improve convergence.
        Uses Kaiming initialization for better gradient flow.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights with Kaiming initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        Forward pass of MSAE

        Parameters:
            x (torch.Tensor): Input HRRP data [batch_size, input_dim]

        Returns:
            torch.Tensor: Reconstructed HRRP data [batch_size, input_dim]
            torch.Tensor: Latent representation [batch_size, latent_dim]
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def get_weight_loss(self):
        """
        Calculate the L2 regularization term for weights

        Returns:
            torch.Tensor: Weight regularization loss
        """
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += torch.sum(param ** 2)
        return self.reg_lambda * reg_loss

    def get_sparsity_loss(self, latent_batch):
        """
        Calculate the sparsity penalty term using KL divergence

        Parameters:
            latent_batch (torch.Tensor): Batch of latent representations

        Returns:
            torch.Tensor: Sparsity loss
        """
        # Average activation of each neuron across the batch
        rho_hat = torch.mean(latent_batch, dim=0)

        # KL divergence between sparsity_param and rho_hat
        kl_div = self.sparsity_param * torch.log(self.sparsity_param / (rho_hat + 1e-10)) + \
                 (1 - self.sparsity_param) * torch.log((1 - self.sparsity_param) / (1 - rho_hat + 1e-10))

        # Additional variance regularization term
        latent_var = torch.var(latent_batch, dim=0)
        var_term = self.sparsity_param * (1 - self.sparsity_param) * torch.log(
            (self.sparsity_param * (1 - self.sparsity_param)) / (latent_var + 1e-10)
        )

        # Complete sparsity loss
        sparsity_loss = self.sparsity_beta * (torch.sum(kl_div) + torch.sum(var_term))
        return sparsity_loss

    def modify_weights_with_svd(self, threshold=0.1):
        """
        Apply SVD-based weight modification to suppress noise
        Sets singular values below the threshold to zero

        Parameters:
            threshold (float): Threshold for singular value pruning
        """
        with torch.no_grad():
            # Apply SVD weight modification to all linear layers
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    # Get weight tensor and convert to 2D matrix
                    weight = m.weight.data
                    original_shape = weight.shape
                    weight_2d = weight.reshape(original_shape[0], -1)

                    # Apply SVD
                    U, S, V = torch.svd(weight_2d)

                    # Create mask for values above threshold
                    mask = S > (threshold * torch.max(S))
                    S_modified = S * mask

                    # Reconstruct modified weight matrix
                    weight_modified = torch.mm(U * S_modified.unsqueeze(0), V.t())

                    # Update model weights
                    m.weight.data = weight_modified.reshape(original_shape)