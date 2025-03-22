# test_all.py
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from pathlib import Path

from models.modules import TargetRadialLengthModule, TargetIdentityModule
from models.cgan_models import Generator
from models.cae_models import ConvAutoEncoder
from models.ae_models import AutoEncoder
from utils.hrrp_dataset import HRRPDataset
from torch.utils.data import DataLoader
from utils.noise_utils import add_noise_for_exact_psnr, calculate_psnr, calculate_ssim

# Set matplotlib parameters for better visualizations
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Define color scheme for visualizations
COLORS = {
    'noisy': '#7F7F7F',  # Gray
    'cgan': '#1F77B4',  # Blue
    'cae': '#2CA02C',  # Green
    'ae': '#D62728',  # Red
    'clean': '#000000'  # Black
}

def test_cgan(args, device, psnr_level):
    """
    Test CGAN model for HRRP signal denoising at a specific PSNR level

    Args:
        args: Testing arguments
        device: Device to test on (CPU or GPU)
        psnr_level: Target PSNR level in dB

    Returns:
        Dictionary of test metrics
    """
    print(f"Testing CGAN for PSNR level {psnr_level}dB...")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f"cgan_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # Load feature extractors
    G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)
    G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                               num_classes=args.num_classes).to(device)

    # Load generator
    generator = Generator(input_dim=args.input_dim,
                          condition_dim=args.feature_dim * 2,
                          hidden_dim=args.hidden_dim).to(device)

    # Load model weights
    cgan_dir = os.path.join(args.load_dir, f"cgan_psnr_{psnr_level}dB")

    # Check if model exists
    if not os.path.exists(cgan_dir):
        print(f"No CGAN model found for PSNR={psnr_level}dB at {cgan_dir}")
        return None

    # Load model weights
    G_D.load_state_dict(torch.load(os.path.join(cgan_dir, 'G_D_final.pth')))
    G_I.load_state_dict(torch.load(os.path.join(cgan_dir, 'G_I_final.pth')))
    generator.load_state_dict(torch.load(os.path.join(cgan_dir, 'generator_final.pth')))

    # Set models to evaluation mode
    G_D.eval()
    G_I.eval()
    generator.eval()

    # Load test dataset
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluation metrics
    mse_loss = nn.MSELoss()
    total_noisy_mse = 0
    total_denoised_mse = 0
    total_noisy_psnr = 0
    total_denoised_psnr = 0
    total_denoised_ssim = 0

    # Test samples with progress bar
    progress_bar = tqdm(range(min(args.num_samples, len(test_loader))), desc=f"Testing CGAN PSNR={psnr_level}dB")

    results = []

    with torch.no_grad():
        for i, (clean_data, radial_length, identity_label) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # Move data to device
            clean_data = clean_data.float().to(device)

            # Create noisy data at the target PSNR
            noisy_data, actual_psnr = add_noise_for_exact_psnr(clean_data, psnr_level)

            # Extract features and create condition
            f_D, _ = G_D(clean_data)
            f_I, _ = G_I(clean_data)
            condition = torch.cat([f_D, f_I], dim=1)

            # Generate denoised data
            denoised_data = generator(noisy_data, condition)

            # Calculate metrics
            noisy_mse = mse_loss(noisy_data, clean_data).item()
            denoised_mse = mse_loss(denoised_data, clean_data).item()

            noisy_psnr = calculate_psnr(clean_data, noisy_data)
            denoised_psnr = calculate_psnr(clean_data, denoised_data)

            # Calculate SSIM (convert tensors to numpy arrays)
            clean_np = clean_data.cpu().numpy()[0]
            denoised_np = denoised_data.cpu().numpy()[0]
            denoised_ssim = calculate_ssim(clean_np, denoised_np)

            # Accumulate metrics
            total_noisy_mse += noisy_mse
            total_denoised_mse += denoised_mse
            total_noisy_psnr += noisy_psnr
            total_denoised_psnr += denoised_psnr
            total_denoised_ssim += denoised_ssim

            # Store individual results
            results.append({
                'sample_idx': i,
                'noisy_mse': noisy_mse,
                'denoised_mse': denoised_mse,
                'noisy_psnr': noisy_psnr,
                'denoised_psnr': denoised_psnr,
                'denoised_ssim': denoised_ssim,
                'psnr_improvement': denoised_psnr - noisy_psnr
            })

            # Plot and save results
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(clean_data.cpu().numpy()[0], color=COLORS['clean'], linewidth=1.5)
            plt.title('Clean HRRP')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.subplot(1, 3, 2)
            plt.plot(noisy_data.cpu().numpy()[0], color=COLORS['noisy'], linewidth=1.5)
            plt.title(f'Noisy HRRP (PSNR: {noisy_psnr:.2f}dB)')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.subplot(1, 3, 3)
            plt.plot(denoised_data.cpu().numpy()[0], color=COLORS['cgan'], linewidth=1.5)
            plt.title(f'CGAN Denoised (PSNR: {denoised_psnr:.2f}dB, SSIM: {denoised_ssim:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i + 1}_denoising.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'Noisy PSNR': f"{noisy_psnr:.2f}dB",
                'Denoised PSNR': f"{denoised_psnr:.2f}dB",
                'Improvement': f"{denoised_psnr - noisy_psnr:.2f}dB"
            })

    # Calculate average metrics
    n_samples = min(args.num_samples, len(test_loader))
    avg_noisy_mse = total_noisy_mse / n_samples
    avg_denoised_mse = total_denoised_mse / n_samples
    avg_noisy_psnr = total_noisy_psnr / n_samples
    avg_denoised_psnr = total_denoised_psnr / n_samples
    avg_denoised_ssim = total_denoised_ssim / n_samples
    avg_psnr_improvement = avg_denoised_psnr - avg_noisy_psnr

    # Save summary metrics
    summary = {
        'model': 'CGAN',
        'psnr_level': psnr_level,
        'avg_noisy_mse': avg_noisy_mse,
        'avg_denoised_mse': avg_denoised_mse,
        'avg_noisy_psnr': avg_noisy_psnr,
        'avg_denoised_psnr': avg_denoised_psnr,
        'avg_denoised_ssim': avg_denoised_ssim,
        'avg_psnr_improvement': avg_psnr_improvement,
        'individual_results': results
    }

    # Print summary
    print(f"\nCGAN Results for PSNR={psnr_level}dB:")
    print(f"  Average Noisy PSNR: {avg_noisy_psnr:.2f}dB")
    print(f"  Average Denoised PSNR: {avg_denoised_psnr:.2f}dB")
    print(f"  Average PSNR Improvement: {avg_psnr_improvement:.2f}dB")
    print(f"  Average Denoised SSIM: {avg_denoised_ssim:.4f}")
    print(f"  Average Noisy MSE: {avg_noisy_mse:.6f}")
    print(f"  Average Denoised MSE: {avg_denoised_mse:.6f}")

    # Write summary to file
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"CGAN Test Results for PSNR={psnr_level}dB\n")
        f.write(f"======================================\n\n")
        f.write(f"Number of test samples: {n_samples}\n\n")
        f.write(f"Average Metrics:\n")
        f.write(f"  Noisy PSNR: {avg_noisy_psnr:.2f}dB\n")
        f.write(f"  Denoised PSNR: {avg_denoised_psnr:.2f}dB\n")
        f.write(f"  PSNR Improvement: {avg_psnr_improvement:.2f}dB\n")
        f.write(f"  Denoised SSIM: {avg_denoised_ssim:.4f}\n")
        f.write(f"  Noisy MSE: {avg_noisy_mse:.6f}\n")
        f.write(f"  Denoised MSE: {avg_denoised_mse:.6f}\n\n")

        f.write(f"Individual Sample Results:\n")
        for res in results:
            f.write(f"  Sample {res['sample_idx'] + 1}:\n")
            f.write(f"    Noisy PSNR: {res['noisy_psnr']:.2f}dB\n")
            f.write(f"    Denoised PSNR: {res['denoised_psnr']:.2f}dB\n")
            f.write(f"    PSNR Improvement: {res['psnr_improvement']:.2f}dB\n")
            f.write(f"    Denoised SSIM: {res['denoised_ssim']:.4f}\n")
            f.write(f"    Noisy MSE: {res['noisy_mse']:.6f}\n")
            f.write(f"    Denoised MSE: {res['denoised_mse']:.6f}\n\n")

    return summary


def test_cae(args, device, psnr_level):
    """
    Test CAE model for HRRP signal denoising at a specific PSNR level

    Args:
        args: Testing arguments
        device: Device to test on (CPU or GPU)
        psnr_level: Target PSNR level in dB

    Returns:
        Dictionary of test metrics
    """
    print(f"Testing CAE for PSNR level {psnr_level}dB...")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f"cae_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # Load CAE model
    model = ConvAutoEncoder(input_dim=args.input_dim,
                            latent_dim=args.latent_dim,
                            hidden_dim=args.hidden_dim).to(device)

    # Load model weights
    cae_dir = os.path.join(args.load_dir, f"cae_psnr_{psnr_level}dB")

    # Check if model exists
    if not os.path.exists(cae_dir):
        print(f"No CAE model found for PSNR={psnr_level}dB at {cae_dir}")
        return None

    # Load model weights
    model.load_state_dict(torch.load(os.path.join(cae_dir, 'cae_model_final.pth')))

    # Set model to evaluation mode
    model.eval()

    # Load test dataset
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluation metrics
    mse_loss = nn.MSELoss()
    total_noisy_mse = 0
    total_denoised_mse = 0
    total_noisy_psnr = 0
    total_denoised_psnr = 0
    total_denoised_ssim = 0

    # Test samples with progress bar
    progress_bar = tqdm(range(min(args.num_samples, len(test_loader))), desc=f"Testing CAE PSNR={psnr_level}dB")

    results = []

    with torch.no_grad():
        for i, (clean_data, _, _) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # Move data to device
            clean_data = clean_data.float().to(device)

            # Create noisy data at the target PSNR
            noisy_data, actual_psnr = add_noise_for_exact_psnr(clean_data, psnr_level)

            # Generate denoised data
            denoised_data, _ = model(noisy_data)

            # Calculate metrics
            noisy_mse = mse_loss(noisy_data, clean_data).item()
            denoised_mse = mse_loss(denoised_data, clean_data).item()

            noisy_psnr = calculate_psnr(clean_data, noisy_data)
            denoised_psnr = calculate_psnr(clean_data, denoised_data)

            # Calculate SSIM (convert tensors to numpy arrays)
            clean_np = clean_data.cpu().numpy()[0]
            denoised_np = denoised_data.cpu().numpy()[0]
            denoised_ssim = calculate_ssim(clean_np, denoised_np)

            # Accumulate metrics
            total_noisy_mse += noisy_mse
            total_denoised_mse += denoised_mse
            total_noisy_psnr += noisy_psnr
            total_denoised_psnr += denoised_psnr
            total_denoised_ssim += denoised_ssim

            # Store individual results
            results.append({
                'sample_idx': i,
                'noisy_mse': noisy_mse,
                'denoised_mse': denoised_mse,
                'noisy_psnr': noisy_psnr,
                'denoised_psnr': denoised_psnr,
                'denoised_ssim': denoised_ssim,
                'psnr_improvement': denoised_psnr - noisy_psnr
            })

            # Plot and save results
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(clean_data.cpu().numpy()[0], color=COLORS['clean'], linewidth=1.5)
            plt.title('Clean HRRP')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.subplot(1, 3, 2)
            plt.plot(noisy_data.cpu().numpy()[0], color=COLORS['noisy'], linewidth=1.5)
            plt.title(f'Noisy HRRP (PSNR: {noisy_psnr:.2f}dB)')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.subplot(1, 3, 3)
            plt.plot(denoised_data.cpu().numpy()[0], color=COLORS['cae'], linewidth=1.5)
            plt.title(f'CAE Denoised (PSNR: {denoised_psnr:.2f}dB, SSIM: {denoised_ssim:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i + 1}_denoising.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'Noisy PSNR': f"{noisy_psnr:.2f}dB",
                'Denoised PSNR': f"{denoised_psnr:.2f}dB",
                'Improvement': f"{denoised_psnr - noisy_psnr:.2f}dB"
            })

    # Calculate average metrics
    n_samples = min(args.num_samples, len(test_loader))
    avg_noisy_mse = total_noisy_mse / n_samples
    avg_denoised_mse = total_denoised_mse / n_samples
    avg_noisy_psnr = total_noisy_psnr / n_samples
    avg_denoised_psnr = total_denoised_psnr / n_samples
    avg_denoised_ssim = total_denoised_ssim / n_samples
    avg_psnr_improvement = avg_denoised_psnr - avg_noisy_psnr

    # Save summary metrics
    summary = {
        'model': 'CAE',
        'psnr_level': psnr_level,
        'avg_noisy_mse': avg_noisy_mse,
        'avg_denoised_mse': avg_denoised_mse,
        'avg_noisy_psnr': avg_noisy_psnr,
        'avg_denoised_psnr': avg_denoised_psnr,
        'avg_denoised_ssim': avg_denoised_ssim,
        'avg_psnr_improvement': avg_psnr_improvement,
        'individual_results': results
    }

    # Print summary
    print(f"\nCAE Results for PSNR={psnr_level}dB:")
    print(f"  Average Noisy PSNR: {avg_noisy_psnr:.2f}dB")
    print(f"  Average Denoised PSNR: {avg_denoised_psnr:.2f}dB")
    print(f"  Average PSNR Improvement: {avg_psnr_improvement:.2f}dB")
    print(f"  Average Denoised SSIM: {avg_denoised_ssim:.4f}")
    print(f"  Average Noisy MSE: {avg_noisy_mse:.6f}")
    print(f"  Average Denoised MSE: {avg_denoised_mse:.6f}")

    # Write summary to file
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"CAE Test Results for PSNR={psnr_level}dB\n")
        f.write(f"====================================\n\n")
        f.write(f"Number of test samples: {n_samples}\n\n")
        f.write(f"Average Metrics:\n")
        f.write(f"  Noisy PSNR: {avg_noisy_psnr:.2f}dB\n")
        f.write(f"  Denoised PSNR: {avg_denoised_psnr:.2f}dB\n")
        f.write(f"  PSNR Improvement: {avg_psnr_improvement:.2f}dB\n")
        f.write(f"  Denoised SSIM: {avg_denoised_ssim:.4f}\n")
        f.write(f"  Noisy MSE: {avg_noisy_mse:.6f}\n")
        f.write(f"  Denoised MSE: {avg_denoised_mse:.6f}\n\n")

        f.write(f"Individual Sample Results:\n")
        for res in results:
            f.write(f"  Sample {res['sample_idx'] + 1}:\n")
            f.write(f"    Noisy PSNR: {res['noisy_psnr']:.2f}dB\n")
            f.write(f"    Denoised PSNR: {res['denoised_psnr']:.2f}dB\n")
            f.write(f"    PSNR Improvement: {res['psnr_improvement']:.2f}dB\n")
            f.write(f"    Denoised SSIM: {res['denoised_ssim']:.4f}\n")
            f.write(f"    Noisy MSE: {res['noisy_mse']:.6f}\n")
            f.write(f"    Denoised MSE: {res['denoised_mse']:.6f}\n\n")

    return summary


def test_ae(args, device, psnr_level):
    """
    Test AE model for HRRP signal denoising at a specific PSNR level

    Args:
        args: Testing arguments
        device: Device to test on (CPU or GPU)
        psnr_level: Target PSNR level in dB

    Returns:
        Dictionary of test metrics
    """
    print(f"Testing AE for PSNR level {psnr_level}dB...")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f"ae_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # Load AE model
    model = AutoEncoder(input_dim=args.input_dim,
                        latent_dim=args.latent_dim,
                        hidden_dim=args.ae_hidden_dim).to(device)

    # Load model weights
    ae_dir = os.path.join(args.load_dir, f"ae_psnr_{psnr_level}dB")

    # Check if model exists
    if not os.path.exists(ae_dir):
        print(f"No AE model found for PSNR={psnr_level}dB at {ae_dir}")
        return None

    # Load model weights
    model.load_state_dict(torch.load(os.path.join(ae_dir, 'ae_model_final.pth')))

    # Set model to evaluation mode
    model.eval()

    # Load test dataset
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluation metrics
    mse_loss = nn.MSELoss()
    total_noisy_mse = 0
    total_denoised_mse = 0
    total_noisy_psnr = 0
    total_denoised_psnr = 0
    total_denoised_ssim = 0

    # Test samples with progress bar
    progress_bar = tqdm(range(min(args.num_samples, len(test_loader))), desc=f"Testing AE PSNR={psnr_level}dB")

    results = []

    with torch.no_grad():
        for i, (clean_data, _, _) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # Move data to device
            clean_data = clean_data.float().to(device)

            # Create noisy data at the target PSNR
            noisy_data, actual_psnr = add_noise_for_exact_psnr(clean_data, psnr_level)

            # Generate denoised data
            denoised_data, _ = model(noisy_data)

            # Calculate metrics
            noisy_mse = mse_loss(noisy_data, clean_data).item()
            denoised_mse = mse_loss(denoised_data, clean_data).item()

            noisy_psnr = calculate_psnr(clean_data, noisy_data)
            denoised_psnr = calculate_psnr(clean_data, denoised_data)

            # Calculate SSIM (convert tensors to numpy arrays)
            clean_np = clean_data.cpu().numpy()[0]
            denoised_np = denoised_data.cpu().numpy()[0]
            denoised_ssim = calculate_ssim(clean_np, denoised_np)

            # Accumulate metrics
            total_noisy_mse += noisy_mse
            total_denoised_mse += denoised_mse
            total_noisy_psnr += noisy_psnr
            total_denoised_psnr += denoised_psnr
            total_denoised_ssim += denoised_ssim

            # Store individual results
            results.append({
                'sample_idx': i,
                'noisy_mse': noisy_mse,
                'denoised_mse': denoised_mse,
                'noisy_psnr': noisy_psnr,
                'denoised_psnr': denoised_psnr,
                'denoised_ssim': denoised_ssim,
                'psnr_improvement': denoised_psnr - noisy_psnr
            })

            # Plot and save results
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(clean_data.cpu().numpy()[0], color=COLORS['clean'], linewidth=1.5)
            plt.title('Clean HRRP')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.subplot(1, 3, 2)
            plt.plot(noisy_data.cpu().numpy()[0], color=COLORS['noisy'], linewidth=1.5)
            plt.title(f'Noisy HRRP (PSNR: {noisy_psnr:.2f}dB)')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.subplot(1, 3, 3)
            plt.plot(denoised_data.cpu().numpy()[0], color=COLORS['ae'], linewidth=1.5)
            plt.title(f'AE Denoised (PSNR: {denoised_psnr:.2f}dB, SSIM: {denoised_ssim:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i + 1}_denoising.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'Noisy PSNR': f"{noisy_psnr:.2f}dB",
                'Denoised PSNR': f"{denoised_psnr:.2f}dB",
                'Improvement': f"{denoised_psnr - noisy_psnr:.2f}dB"
            })

    # Calculate average metrics
    n_samples = min(args.num_samples, len(test_loader))
    avg_noisy_mse = total_noisy_mse / n_samples
    avg_denoised_mse = total_denoised_mse / n_samples
    avg_noisy_psnr = total_noisy_psnr / n_samples
    avg_denoised_psnr = total_denoised_psnr / n_samples
    avg_denoised_ssim = total_denoised_ssim / n_samples
    avg_psnr_improvement = avg_denoised_psnr - avg_noisy_psnr

    # Save summary metrics
    summary = {
        'model': 'AE',
        'psnr_level': psnr_level,
        'avg_noisy_mse': avg_noisy_mse,
        'avg_denoised_mse': avg_denoised_mse,
        'avg_noisy_psnr': avg_noisy_psnr,
        'avg_denoised_psnr': avg_denoised_psnr,
        'avg_denoised_ssim': avg_denoised_ssim,
        'avg_psnr_improvement': avg_psnr_improvement,
        'individual_results': results
    }

    # Print summary
    print(f"\nAE Results for PSNR={psnr_level}dB:")
    print(f"  Average Noisy PSNR: {avg_noisy_psnr:.2f}dB")
    print(f"  Average Denoised PSNR: {avg_denoised_psnr:.2f}dB")
    print(f"  Average PSNR Improvement: {avg_psnr_improvement:.2f}dB")
    print(f"  Average Denoised SSIM: {avg_denoised_ssim:.4f}")
    print(f"  Average Noisy MSE: {avg_noisy_mse:.6f}")
    print(f"  Average Denoised MSE: {avg_denoised_mse:.6f}")

    # Write summary to file
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"AE Test Results for PSNR={psnr_level}dB\n")
        f.write(f"===================================\n\n")
        f.write(f"Number of test samples: {n_samples}\n\n")
        f.write(f"Average Metrics:\n")
        f.write(f"  Noisy PSNR: {avg_noisy_psnr:.2f}dB\n")
        f.write(f"  Denoised PSNR: {avg_denoised_psnr:.2f}dB\n")
        f.write(f"  PSNR Improvement: {avg_psnr_improvement:.2f}dB\n")
        f.write(f"  Denoised SSIM: {avg_denoised_ssim:.4f}\n")
        f.write(f"  Noisy MSE: {avg_noisy_mse:.6f}\n")
        f.write(f"  Denoised MSE: {avg_denoised_mse:.6f}\n\n")

        f.write(f"Individual Sample Results:\n")
        for res in results:
            f.write(f"  Sample {res['sample_idx'] + 1}:\n")
            f.write(f"    Noisy PSNR: {res['noisy_psnr']:.2f}dB\n")
            f.write(f"    Denoised PSNR: {res['denoised_psnr']:.2f}dB\n")
            f.write(f"    PSNR Improvement: {res['psnr_improvement']:.2f}dB\n")
            f.write(f"    Denoised SSIM: {res['denoised_ssim']:.4f}\n")
            f.write(f"    Noisy MSE: {res['noisy_mse']:.6f}\n")
            f.write(f"    Denoised MSE: {res['denoised_mse']:.6f}\n\n")

    return summary


def compare_methods(args, psnr_level, results):
    """
    Compare all denoising methods at a specific PSNR level and create visualizations

    Args:
        args: Testing arguments
        psnr_level: PSNR level in dB
        results: Dictionary of results from all models
    """
    print(f"Generating comparison visualizations for PSNR={psnr_level}dB...")

    # Create output directory for comparisons
    output_dir = os.path.join(args.output_dir, f"comparison_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # Extract results
    methods = list(results.keys())

    # Create bar plots comparing metrics

    # 1. PSNR Improvement plot
    plt.figure(figsize=(10, 6))

    psnr_improvements = [results[method]['avg_psnr_improvement'] for method in methods]
    method_colors = [COLORS[method.lower()] for method in methods]

    bars = plt.bar(methods, psnr_improvements, color=method_colors, width=0.6, edgecolor='k', linewidth=1)

    plt.title(f'PSNR Improvement Comparison at {psnr_level}dB Noise Level', pad=15)
    plt.xlabel('Methods')
    plt.ylabel('PSNR Improvement (dB)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.spines['top'].set_visible(False)
    plt.spines['right'].set_visible(False)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'+{height:.2f}dB', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_improvement_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. SSIM Comparison plot
    plt.figure(figsize=(10, 6))

    ssim_values = [results[method]['avg_denoised_ssim'] for method in methods]

    bars = plt.bar(methods, ssim_values, color=method_colors, width=0.6, edgecolor='k', linewidth=1)

    plt.title(f'SSIM Comparison at {psnr_level}dB Noise Level', pad=15)
    plt.xlabel('Methods')
    plt.ylabel('Structural Similarity Index (SSIM)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.spines['top'].set_visible(False)
    plt.spines['right'].set_visible(False)

    # Set y-axis to start from a reasonable value to better show differences
    ssim_min = min(ssim_values) * 0.95
    plt.ylim(ssim_min, 1.0)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. MSE Comparison plot
    plt.figure(figsize=(10, 6))

    mse_values = [results[method]['avg_denoised_mse'] for method in methods]

    bars = plt.bar(methods, mse_values, color=method_colors, width=0.6, edgecolor='k', linewidth=1)

    plt.title(f'MSE Comparison at {psnr_level}dB Noise Level', pad=15)
    plt.xlabel('Methods')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.spines['top'].set_visible(False)
    plt.spines['right'].set_visible(False)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.0005,
                 f'{height:.6f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Write comparison report
    with open(os.path.join(output_dir, 'comparison_results.txt'), 'w') as f:
        f.write(f"Denoising Methods Comparison at PSNR={psnr_level}dB\n")
        f.write(f"=================================================\n\n")

        # Table header
        f.write(f"{'Method':<10} {'Denoised PSNR':<15} {'PSNR Gain':<15} {'SSIM':<15} {'MSE':<15}\n")
        f.write(f"{'-' * 60}\n")

        # Table rows
        for method in methods:
            res = results[method]
            f.write(f"{method:<10} {res['avg_denoised_psnr']:<15.2f} {res['avg_psnr_improvement']:<15.2f} "
                    f"{res['avg_denoised_ssim']:<15.4f} {res['avg_denoised_mse']:<15.6f}\n")

        f.write(f"\n")

        # Find best method for each metric
        best_psnr = max(methods, key=lambda m: results[m]['avg_denoised_psnr'])
        best_improve = max(methods, key=lambda m: results[m]['avg_psnr_improvement'])
        best_ssim = max(methods, key=lambda m: results[m]['avg_denoised_ssim'])
        best_mse = min(methods, key=lambda m: results[m]['avg_denoised_mse'])

        f.write(f"Best method by PSNR: {best_psnr} ({results[best_psnr]['avg_denoised_psnr']:.2f}dB)\n")
        f.write(
            f"Best method by PSNR improvement: {best_improve} (+{results[best_improve]['avg_psnr_improvement']:.2f}dB)\n")
        f.write(f"Best method by SSIM: {best_ssim} ({results[best_ssim]['avg_denoised_ssim']:.4f})\n")
        f.write(f"Best method by MSE: {best_mse} ({results[best_mse]['avg_denoised_mse']:.6f})\n")

    print(f"Comparison results saved to {output_dir}")


def generate_psnr_comparison(args, all_results):
    """
    Generate comparison charts across different PSNR levels

    Args:
        args: Testing arguments
        all_results: Dictionary of results from all models and PSNR levels
    """
    print("Generating PSNR level comparison visualizations...")

    # Create output directory
    output_dir = os.path.join(args.output_dir, "psnr_comparison")
    os.makedirs(output_dir, exist_ok=True)

    # Extract PSNR levels and methods
    psnr_levels = sorted(list(all_results.keys()))
    methods = list(all_results[psnr_levels[0]].keys())

    # 1. PSNR Improvement vs. Noise Level
    plt.figure(figsize=(12, 6))

    width = 0.25
    x = np.arange(len(psnr_levels))

    for i, method in enumerate(methods):
        improvements = [all_results[psnr][method]['avg_psnr_improvement'] for psnr in psnr_levels]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = plt.bar(x + offset, improvements, width, label=method, color=COLORS[method.lower()], edgecolor='k',
                       linewidth=1)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'+{height:.1f}dB', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Input Noise Level (PSNR)')
    plt.ylabel('PSNR Improvement (dB)')
    plt.title('PSNR Improvement vs. Noise Level', pad=15)
    plt.xticks(x, [f'{level}dB' for level in psnr_levels])
    plt.legend(frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.spines['top'].set_visible(False)
    plt.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_improvement_vs_noise.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. SSIM vs. Noise Level
    plt.figure(figsize=(12, 6))

    for i, method in enumerate(methods):
        ssim_values = [all_results[psnr][method]['avg_denoised_ssim'] for psnr in psnr_levels]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = plt.bar(x + offset, ssim_values, width, label=method, color=COLORS[method.lower()], edgecolor='k',
                       linewidth=1)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Input Noise Level (PSNR)')
    plt.ylabel('Average SSIM')
    plt.title('SSIM vs. Noise Level', pad=15)
    plt.xticks(x, [f'{level}dB' for level in psnr_levels])
    plt.legend(frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.spines['top'].set_visible(False)
    plt.spines['right'].set_visible(False)

    # Set SSIM axis to better show differences
    ssim_min = min([all_results[psnr][method]['avg_denoised_ssim']
                    for psnr in psnr_levels for method in methods]) * 0.95
    plt.ylim(ssim_min, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_vs_noise.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Write summary report
    with open(os.path.join(output_dir, 'summary_results.txt'), 'w') as f:
        f.write(f"Denoising Methods Comparison Across PSNR Levels\n")
        f.write(f"==============================================\n\n")

        # For each PSNR level
        for psnr in psnr_levels:
            f.write(f"PSNR = {psnr}dB Results:\n")
            f.write(f"{'-' * 40}\n")

            # Table header
            f.write(f"{'Method':<10} {'Denoised PSNR':<15} {'PSNR Gain':<15} {'SSIM':<15}\n")
            f.write(f"{'-' * 55}\n")

            # Table rows
            for method in methods:
                res = all_results[psnr][method]
                f.write(f"{method:<10} {res['avg_denoised_psnr']:<15.2f} {res['avg_psnr_improvement']:<15.2f} "
                        f"{res['avg_denoised_ssim']:<15.4f}\n")

            # Find best methods for this PSNR level
            best_psnr = max(methods, key=lambda m: all_results[psnr][m]['avg_denoised_psnr'])
            best_improve = max(methods, key=lambda m: all_results[psnr][m]['avg_psnr_improvement'])
            best_ssim = max(methods, key=lambda m: all_results[psnr][m]['avg_denoised_ssim'])

            f.write(f"\nBest method by PSNR: {best_psnr}\n")
            f.write(f"Best method by PSNR improvement: {best_improve}\n")
            f.write(f"Best method by SSIM: {best_ssim}\n\n")
            f.write(f"{'=' * 60}\n\n")

        # Count how many times each method wins
        f.write("Summary of Best Performing Methods:\n")
        f.write(f"{'-' * 40}\n\n")

        psnr_winners = {}
        ssim_winners = {}

        for psnr in psnr_levels:
            best_psnr = max(methods, key=lambda m: all_results[psnr][m]['avg_psnr_improvement'])
            best_ssim = max(methods, key=lambda m: all_results[psnr][m]['avg_denoised_ssim'])

            psnr_winners[best_psnr] = psnr_winners.get(best_psnr, 0) + 1
            ssim_winners[best_ssim] = ssim_winners.get(best_ssim, 0) + 1

        f.write("Best method by PSNR improvement:\n")
        for method, count in psnr_winners.items():
            f.write(f"  {method}: {count}/{len(psnr_levels)} times\n")

        f.write("\nBest method by SSIM:\n")
        for method, count in ssim_winners.items():
            f.write(f"  {method}: {count}/{len(psnr_levels)} times\n")

    print(f"PSNR comparison results saved to {output_dir}")


def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Unified testing script for HRRP denoising models')

    # General parameters
    parser.add_argument('--model', type=str, default='all',
                        choices=['cgan', 'cae', 'ae', 'all'],
                        help='Model to test')
    parser.add_argument('--test_dir', type=str, default='datasets/simulated_3/test',
                        help='Directory containing test data')
    parser.add_argument('--load_dir', type=str, default='checkpoints',
                        help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save test results')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test samples to process')
    parser.add_argument('--input_dim', type=int, default=500,
                        help='Dimension of input HRRP sequence')
    parser.add_argument('--psnr_levels', type=str, default='20,10,0',
                        help='PSNR levels to test at (comma-separated values in dB)')

    # Feature extractors parameters
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Dimension of feature extractors output')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of target identity classes')

    # CGAN specific parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of hidden layers for CGAN and CAE')

    # CAE and AE specific parameters
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space for CAE and AE')
    parser.add_argument('--ae_hidden_dim', type=int, default=256,
                        help='Dimension of hidden layers for AE')

    args = parser.parse_args()

    # Parse PSNR levels
    psnr_levels = [float(level) for level in args.psnr_levels.split(',')]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Store all results
    all_results = {}

    # Test models at each PSNR level
    for psnr_level in psnr_levels:
        print(f"\n{'=' * 50}")
        print(f"Testing at PSNR level: {psnr_level}dB")
        print(f"{'=' * 50}\n")

        # Results for this PSNR level
        psnr_results = {}

        if args.model in ['cgan', 'all']:
            cgan_result = test_cgan(args, device, psnr_level)
            if cgan_result:
                psnr_results['CGAN'] = cgan_result

        if args.model in ['cae', 'all']:
            cae_result = test_cae(args, device, psnr_level)
            if cae_result:
                psnr_results['CAE'] = cae_result

        if args.model in ['ae', 'all']:
            ae_result = test_ae(args, device, psnr_level)
            if ae_result:
                psnr_results['AE'] = ae_result

        # Compare methods if we have results from multiple models
        if len(psnr_results) > 1:
            compare_methods(args, psnr_level, psnr_results)

        # Store results for this PSNR level
        all_results[psnr_level] = psnr_results

    # Generate comparison across PSNR levels
    if len(all_results) > 1 and all(len(results) > 0 for results in all_results.values()):
        generate_psnr_comparison(args, all_results)

    print("\nTesting complete for all models and PSNR levels.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total testing time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")