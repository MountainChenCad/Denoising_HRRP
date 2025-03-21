# train_cae.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from cae_models import ConvAutoEncoder
from hrrp_dataset import HRRPDataset
import random


def add_noise(hrrp_data, noise_level=0.1):
    """
    Add Gaussian noise to HRRP data

    Parameters:
        hrrp_data (torch.Tensor): Clean HRRP data
        noise_level (float): Standard deviation of Gaussian noise

    Returns:
        torch.Tensor: Noisy HRRP data
    """
    noise = torch.randn_like(hrrp_data) * noise_level
    noisy_data = hrrp_data + noise
    # Ensure data stays in valid range [0, 1]
    noisy_data = torch.clamp(noisy_data, 0, 1)
    return noisy_data


def train_cae(args):
    """
    Train a CAE for HRRP signal denoising

    Parameters:
        args: Training parameters
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create CAE model
    model = ConvAutoEncoder(input_dim=args.input_dim,
                            latent_dim=args.latent_dim,
                            hidden_dim=args.hidden_dim).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load dataset
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Print information about the dataset and model
    sample_data, _, _ = next(iter(train_loader))
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Model overview: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Arrays for tracking loss
    train_losses = []

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for i, (clean_data, _, _) in enumerate(train_loader):
            # Move data to device
            clean_data = clean_data.float().to(device)

            # Create noisy data
            noisy_data = add_noise(clean_data, noise_level=args.noise_level)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass (denoising task: noisy->clean)
            reconstructed, _ = model(noisy_data)

            # Calculate loss (between reconstruction and clean data)
            loss = criterion(reconstructed, clean_data)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update epoch loss
            epoch_loss += loss.item()

            # Print batch progress
            if i % 10 == 0:
                print(f"[Epoch {epoch + 1}/{args.epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[Loss: {loss.item():.4f}]")

        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{args.epochs} - Avg Loss: {epoch_loss:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save model
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'cae_model.pth'))

            # Save sample denoising results
            if args.save_samples:
                model.eval()
                with torch.no_grad():
                    # Get sample from dataset
                    sample_idx = np.random.randint(0, len(train_dataset))
                    sample_clean, _, _ = train_dataset[sample_idx]
                    sample_clean = sample_clean.unsqueeze(0).float().to(device)

                    # Create noisy sample
                    sample_noisy = add_noise(sample_clean, noise_level=args.noise_level)

                    # Denoise the sample
                    sample_denoised, _ = model(sample_noisy)

                    # Plot results
                    plt.figure(figsize=(12, 4))

                    plt.subplot(1, 3, 1)
                    plt.plot(sample_clean.cpu().numpy()[0])
                    plt.title('Clean HRRP')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.subplot(1, 3, 2)
                    plt.plot(sample_noisy.cpu().numpy()[0])
                    plt.title(f'Noisy HRRP (sigma={args.noise_level})')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.subplot(1, 3, 3)
                    plt.plot(sample_denoised.cpu().numpy()[0])
                    plt.title('Denoised HRRP')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.tight_layout()
                    plt.savefig(os.path.join(checkpoint_dir, 'sample_denoising.png'))
                    plt.close()

                model.train()

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'cae_model_final.pth'))

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CAE Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'training_loss.png'))
    plt.close()

    print(f"Training complete. Model saved to {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CAE for HRRP denoising')
    parser.add_argument('--train_dir', type=str, default='datasets/simulated_3/train',
                        help='Directory containing training data')
    parser.add_argument('--save_dir', type=str, default='checkpoints/cae',
                        help='Directory to save trained models')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='Learning rate')
    parser.add_argument('--input_dim', type=int, default=500,
                        help='Dimension of input HRRP sequence')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of hidden layers')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='Standard deviation of Gaussian noise added to clean samples')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Epoch interval for saving checkpoints')
    parser.add_argument('--save_samples', action='store_true',
                        help='Whether to save sample denoising results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Train CAE
    train_cae(args)