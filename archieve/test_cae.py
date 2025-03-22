# test_cae.py
import torch
import torch.nn as nn
import os
import argparse
import matplotlib.pyplot as plt
from models.cae_models import ConvAutoEncoder
from utils.hrrp_dataset import HRRPDataset
from torch.utils.data import DataLoader


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


def test_cae(args):
    """
    Test CAE for HRRP signal denoising

    Parameters:
        args: Testing parameters
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = ConvAutoEncoder(input_dim=args.input_dim,
                            latent_dim=args.latent_dim,
                            hidden_dim=args.hidden_dim).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(os.path.join(args.load_dir, 'cae_model_final.pth')))
    model.eval()

    # Load test dataset
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define loss function for evaluation
    mse_loss = nn.MSELoss()

    # Denoise test samples
    total_mse_noisy = 0
    total_mse_denoised = 0

    with torch.no_grad():
        for i, (clean_data, radial_length, identity_label) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # Move data to device
            clean_data = clean_data.float().to(device)

            # Create noisy data
            noisy_data = add_noise(clean_data, noise_level=args.noise_level)

            # Generate denoised data
            denoised_data, _ = model(noisy_data)

            # Calculate MSE for noisy and denoised data
            mse_noisy = mse_loss(noisy_data, clean_data).item()
            mse_denoised = mse_loss(denoised_data, clean_data).item()

            total_mse_noisy += mse_noisy
            total_mse_denoised += mse_denoised

            # Plot results
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(clean_data.cpu().numpy()[0])
            plt.title('Clean HRRP')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')

            plt.subplot(1, 3, 2)
            plt.plot(noisy_data.cpu().numpy()[0])
            plt.title(f'Noisy HRRP (MSE: {mse_noisy:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')

            plt.subplot(1, 3, 3)
            plt.plot(denoised_data.cpu().numpy()[0])
            plt.title(f'Denoised HRRP (MSE: {mse_denoised:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')

            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'sample_{i + 1}_denoising.png'))
            plt.close()

            print(f"Sample {i + 1}:")
            print(f"  Noisy MSE: {mse_noisy:.4f}")
            print(f"  Denoised MSE: {mse_denoised:.4f}")
            print(f"  Improvement: {(mse_noisy - mse_denoised) / mse_noisy * 100:.2f}%")

    # Calculate average MSE
    avg_mse_noisy = total_mse_noisy / min(args.num_samples, len(test_loader))
    avg_mse_denoised = total_mse_denoised / min(args.num_samples, len(test_loader))

    print(f"\nAverage Noisy MSE: {avg_mse_noisy:.4f}")
    print(f"Average Denoised MSE: {avg_mse_denoised:.4f}")
    print(f"Average Improvement: {(avg_mse_noisy - avg_mse_denoised) / avg_mse_noisy * 100:.2f}%")

    # Save summary results
    with open(os.path.join(args.output_dir, 'results_summary.txt'), 'w') as f:
        f.write(f"CAE Denoising Results\n")
        f.write(f"===================\n\n")
        f.write(f"Noise Level: {args.noise_level}\n")
        f.write(f"Average Noisy MSE: {avg_mse_noisy:.4f}\n")
        f.write(f"Average Denoised MSE: {avg_mse_denoised:.4f}\n")
        f.write(f"Average Improvement: {(avg_mse_noisy - avg_mse_denoised) / avg_mse_noisy * 100:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test CAE for HRRP denoising')
    parser.add_argument('--test_dir', type=str, default='datasets/simulated_3/test',
                        help='Directory containing test data')
    parser.add_argument('--load_dir', type=str, default='checkpoints/cae',
                        help='Directory to load trained model from')
    parser.add_argument('--output_dir', type=str, default='results/cae',
                        help='Directory to save test results')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test samples to process')
    parser.add_argument('--input_dim', type=int, default=500,
                        help='Dimension of input HRRP sequence')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of hidden layers')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='Standard deviation of Gaussian noise added to clean samples')

    args = parser.parse_args()

    # Test CAE
    test_cae(args)