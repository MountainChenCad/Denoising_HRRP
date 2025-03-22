# msae_loss.py
import torch
import torch.nn as nn


class MSAELoss(nn.Module):
    """
    Custom loss function for Modified Sparse AutoEncoder training
    Combines reconstruction error with weight regularization and sparsity penalty
    """

    def __init__(self):
        super(MSAELoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, model, inputs, outputs, latent):
        """
        Calculate the full MSAE loss

        Parameters:
            model (ModifiedSparseAutoEncoder): MSAE model instance
            inputs (torch.Tensor): Input HRRP signals
            outputs (torch.Tensor): Reconstructed HRRP signals
            latent (torch.Tensor): Latent representations

        Returns:
            torch.Tensor: Total loss
            dict: Dictionary of individual loss components
        """
        # 1. Reconstruction error
        rec_loss = self.reconstruction_loss(outputs, inputs)

        # 2. Weight regularization term
        weight_loss = model.get_weight_loss()

        # 3. Sparsity penalty term
        sparsity_loss = model.get_sparsity_loss(latent)

        # 4. Total loss
        total_loss = rec_loss + weight_loss + sparsity_loss

        # Return individual loss components for monitoring
        loss_components = {
            'total': total_loss.item(),
            'reconstruction': rec_loss.item(),
            'weight_reg': weight_loss.item(),
            'sparsity': sparsity_loss.item()
        }

        return total_loss, loss_components