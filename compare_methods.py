# compare_methods.py
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from models import TargetRadialLengthModule, TargetIdentityModule
from cgan_models import Generator
from cae_models import ConvAutoEncoder
from hrrp_dataset import HRRPDataset
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


def compare_methods(args):
    """
    Compare CGAN and CAE methods for HRRP signal denoising

    Parameters:
        args: Comparison parameters
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CGAN models
    # Feature extractors
    G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)
    G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                               num_classes=args.num_classes).to(device)

    # Load their weights
    G_D.load_state_dict(torch.load(os.path.join(args.cgan_dir, 'G_D.pth')))
    G_I.load_state_dict(torch.load(os.path.join(args.cgan_dir, 'G_I.pth')))

    # Set feature extractors to evaluation mode
    G_D.eval()
    G_I.eval()

    # Load the CGAN generator
    cgan_generator = Generator(input_dim=args.input_dim,
                               condition_dim=args.feature_dim * 2,
                               hidden_dim=args.hidden_dim).to(device)

    # Load generator weights
    cgan_generator.load_state_dict(torch.load(os.path.join(args.cgan_dir, 'generator_final.pth')))
    cgan_generator.eval()

    # Load CAE model
    cae_model = ConvAutoEncoder(input_dim=args.input_dim,
                                latent_dim=args.latent_dim,
                                hidden_dim=args.hidden_dim).to(device)

    # Load CAE weights
    cae_model.load_state_dict(torch.load(os.path.join(args.cae_dir, 'cae_model_final.pth')))
    cae_model.eval()

    # Load test dataset
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define loss function for evaluation
    mse_loss = nn.MSELoss()

    # Compare denoising performance
    cgan_total_mse = 0
    cae_total_mse = 0

    with torch.no_grad():
        for i, (clean_data, radial_length, identity_label) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # Move data to device
            clean_data = clean_data.float().to(device)

            # Create noisy data
            noisy_data = add_noise(clean_data, noise_level=args.noise_level)

            # Extract features for CGAN conditioning
            f_D, _ = G_D(clean_data)
            f_I, _ = G_I(clean_data)
            condition = torch.cat([f_D, f_I], dim=1)

            # Generate denoised data using CGAN
            cgan_denoised = cgan_generator(noisy_data, condition)

            # Generate denoised data using CAE
            cae_denoised, _ = cae_model(noisy_data)

            # Calculate MSE for both methods
            cgan_mse = mse_loss(cgan_denoised, clean_data).item()
            cae_mse = mse_loss(cae_denoised, clean_data).item()

            cgan_total_mse += cgan_mse
            cae_total_mse += cae_mse

            # Plot comparison results
            plt.figure(figsize=(15, 4))

            plt.subplot(1, 4, 1)
            plt.plot(clean_data.cpu().numpy()[0])
            plt.title('Clean HRRP')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')

            plt.subplot(1, 4, 2)
            plt.plot(noisy_data.cpu().numpy()[0])
            plt.title(f'Noisy HRRP')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')

            plt.subplot(1, 4, 3)
            plt.plot(cgan_denoised.cpu().numpy()[0])
            plt.title(f'CGAN Denoised (MSE: {cgan_mse:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')

            plt.subplot(1, 4, 4)
            plt.plot(cae_denoised.cpu().numpy()[0])
            plt.title(f'CAE Denoised (MSE: {cae_mse:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')

            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'comparison_sample_{i + 1}.png'))
            plt.close()

            print(f"Sample {i + 1}:")
            print(f"  CGAN MSE: {cgan_mse:.4f}")
            print(f"  CAE MSE: {cae_mse:.4f}")
            if cgan_mse < cae_mse:
                better = "CGAN"
                diff = (cae_mse - cgan_mse) / cae_mse * 100
            else:
                better = "CAE"
                diff = (cgan_mse - cae_mse) / cgan_mse * 100
            print(f"  Better method: {better} by {diff:.2f}%")

    # Calculate average MSE
    avg_cgan_mse = cgan_total_mse / min(args.num_samples, len(test_loader))
    avg_cae_mse = cae_total_mse / min(args.num_samples, len(test_loader))

    print(f"\nAverage CGAN MSE: {avg_cgan_mse:.4f}")
    print(f"Average CAE MSE: {avg_cae_mse:.4f}")

    # Determine overall better method
    if avg_cgan_mse < avg_cae_mse:
        better_method = "CGAN"
        improvement = (avg_cae_mse - avg_cgan_mse) / avg_cae_mse * 100
    else:
        better_method = "CAE"
        improvement = (avg_cgan_mse - avg_cae_mse) / avg_cgan_mse * 100

    print(f"Overall better method: {better_method} by {improvement:.2f}%")

    # Save summary results
    with open(os.path.join(args.output_dir, 'comparison_results.txt'), 'w') as f:
        f.write(f"CGAN vs CAE Comparison Results\n")
        f.write(f"============================\n\n")
        f.write(f"Noise Level: {args.noise_level}\n")
        f.write(f"Number of test samples: {min(args.num_samples, len(test_loader))}\n\n")
        f.write(f"Average CGAN MSE: {avg_cgan_mse:.4f}\n")
        f.write(f"Average CAE MSE: {avg_cae_mse:.4f}\n")
        f.write(f"Overall better method: {better_method} by {improvement:.2f}%\n")

    # Create a bar chart comparing methods
    methods = ['CGAN', 'CAE']
    mse_values = [avg_cgan_mse, avg_cae_mse]

    plt.figure(figsize=(10, 6))
    plt.bar(methods, mse_values, color=['blue', 'green'])
    plt.title('MSE Comparison between CGAN and CAE')
    plt.xlabel('Method')
    plt.ylabel('Average MSE (lower is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels above bars
    for i, v in enumerate(mse_values):
        plt.text(i, v + 0.001, f"{v:.4f}", ha='center')

    plt.savefig(os.path.join(args.output_dir, 'method_comparison.png'))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare CGAN and CAE for HRRP denoising')
    parser.add_argument('--test_dir', type=str, default='datasets/simulated_3/test',
                        help='Directory containing test data')
    parser.add_argument('--cgan_dir', type=str, default='checkpoints/cgan',
                        help='Directory containing trained CGAN models')
    parser.add_argument('--cae_dir', type=str, default='checkpoints/cae',
                        help='Directory containing trained CAE model')
    parser.add_argument('--output_dir', type=str, default='results/comparison',
                        help='Directory to save comparison results')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test samples to process')
    parser.add_argument('--input_dim', type=int, default=500,
                        help='Dimension of input HRRP sequence')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Feature dimension for CGAN')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent dimension for CAE')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for both models')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of target identity classes')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='Standard deviation of Gaussian noise')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Compare methods
    compare_methods(args)