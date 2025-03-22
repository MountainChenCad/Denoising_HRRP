# noise_utils.py
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def add_noise(hrrp_data, noise_level=0.1):
    """
    Add Gaussian noise of specified intensity to HRRP data

    Parameters:
        hrrp_data (torch.Tensor): Clean HRRP data
        noise_level (float): Standard deviation of Gaussian noise

    Returns:
        torch.Tensor: Noisy HRRP data
    """
    noise = torch.randn_like(hrrp_data) * noise_level
    noisy_data = hrrp_data + noise
    # Ensure data stays within valid range [0, 1]
    noisy_data = torch.clamp(noisy_data, 0, 1)
    return noisy_data


def add_noise_for_psnr(signal, target_psnr_db):
    """
    Add Gaussian noise to achieve a specific PSNR (Peak Signal-to-Noise Ratio) value

    Parameters:
        signal (torch.Tensor): Original clean signal
        target_psnr_db (float): Desired PSNR value (dB)

    Returns:
        torch.Tensor: Signal with added noise
    """
    # Calculate signal power
    signal_power = torch.mean(signal ** 2)

    # Calculate noise power based on desired PSNR
    noise_power = signal_power / (10 ** (target_psnr_db / 10))

    # Generate Gaussian noise
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)

    # Add noise to signal
    noisy_signal = signal + noise

    # Ensure data stays within valid range [0, 1]
    noisy_signal = torch.clamp(noisy_signal, 0, 1)

    return noisy_signal


def add_noise_for_exact_psnr(signal, target_psnr_db, max_iterations=5, tolerance=0.1):
    """
    Add Gaussian noise to precisely achieve a specific PSNR (Peak Signal-to-Noise Ratio) value

    Iteratively adjusts noise intensity to ensure target PSNR is achieved

    Parameters:
        signal (torch.Tensor): Original clean signal
        target_psnr_db (float): Target PSNR (dB)
        max_iterations (int): Maximum number of adjustment iterations
        tolerance (float): Acceptable PSNR error range (dB)

    Returns:
        torch.Tensor: Signal with added noise
        float: Actual measured PSNR
    """
    # Initial estimate
    signal_power = torch.mean(signal ** 2)
    noise_power = signal_power / (10 ** (target_psnr_db / 10))

    best_noisy_signal = None
    best_psnr_diff = float('inf')

    for iteration in range(max_iterations):
        # Generate Gaussian noise
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)

        # Add noise and clip
        noisy_signal = signal + noise
        noisy_signal = torch.clamp(noisy_signal, 0, 1)

        # Calculate actual PSNR
        actual_psnr = calculate_psnr(signal, noisy_signal)
        psnr_diff = abs(actual_psnr - target_psnr_db)

        # Save best result
        if psnr_diff < best_psnr_diff:
            best_noisy_signal = noisy_signal.clone()
            best_psnr_diff = psnr_diff

        # Exit if close enough
        if psnr_diff <= tolerance:
            return noisy_signal, actual_psnr

        # Adjust noise power based on current result
        adjustment_factor = 10 ** ((actual_psnr - target_psnr_db) / 20)
        noise_power = noise_power * adjustment_factor

    # Return result closest to target PSNR
    actual_psnr = calculate_psnr(signal, best_noisy_signal)
    return best_noisy_signal, actual_psnr


def calculate_psnr(original, processed):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio)

    Parameters:
        original (torch.Tensor): Original clean signal
        processed (torch.Tensor): Noisy or reconstructed signal

    Returns:
        float: PSNR value (dB)
    """
    mse = torch.mean((original - processed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(x, y):
    """
    Calculate SSIM (Structural Similarity Index) between two 1D signals

    Parameters:
        x (numpy.ndarray): First signal (1D)
        y (numpy.ndarray): Second signal (1D)

    Returns:
        float: SSIM value between 0 and 1 (higher is better)
    """
    # SSIM requires non-negative inputs
    return ssim(x, y, data_range=1.0)


def psnr_to_noise_level(signal, target_psnr_db):
    """
    Convert target PSNR to corresponding noise level (standard deviation)

    Parameters:
        signal (torch.Tensor): Original clean signal
        target_psnr_db (float): Target PSNR value (dB)

    Returns:
        float: Corresponding noise level (standard deviation)
    """
    signal_power = torch.mean(signal ** 2)
    noise_power = signal_power / (10 ** (target_psnr_db / 10))
    noise_level = torch.sqrt(noise_power).item()
    return noise_level