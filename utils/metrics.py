# metrics.py - Evaluation metrics calculation
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import stats


def calculate_psnr(original, processed):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)

    Parameters:
        original (torch.Tensor): Original clean signal
        processed (torch.Tensor): Processed signal (noisy or restored)

    Returns:
        float: PSNR value (dB)
    """
    if isinstance(original, np.ndarray):
        original = torch.from_numpy(original)
    if isinstance(processed, np.ndarray):
        processed = torch.from_numpy(processed)

    # Ensure input tensors are float type
    original = original.float()
    processed = processed.float()

    # Calculate MSE
    mse = torch.mean((original - processed) ** 2)
    if mse == 0:
        return 100.0

    # Assume signal amplitude range is [0, 1]
    max_val = 1.0
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))

    return psnr.item()


def calculate_ssim(x, y):
    """
    Calculate Structural Similarity Index (SSIM)

    Parameters:
        x (numpy.ndarray): First signal
        y (numpy.ndarray): Second signal

    Returns:
        float: SSIM value [0,1] higher is better
    """
    # Ensure inputs are numpy arrays
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # If batch data, take the first sample
    if x.ndim > 1 and x.shape[0] == 1:
        x = x[0]
    if y.ndim > 1 and y.shape[0] == 1:
        y = y[0]

    # SSIM requires non-negative inputs
    return ssim(x, y, data_range=1.0)


def calculate_mse(original, processed):
    """
    Calculate Mean Squared Error (MSE)

    Parameters:
        original (torch.Tensor or numpy.ndarray): Original clean signal
        processed (torch.Tensor or numpy.ndarray): Processed signal

    Returns:
        float: MSE value
    """
    if isinstance(original, np.ndarray):
        original = torch.from_numpy(original).float()
    if isinstance(processed, np.ndarray):
        processed = torch.from_numpy(processed).float()

    # Ensure input tensors are float type
    original = original.float()
    processed = processed.float()

    # Calculate MSE
    mse = torch.mean((original - processed) ** 2)

    return mse.item()


def calculate_improvement(noisy_metric, denoised_metric, higher_is_better=True):
    """
    Calculate denoising improvement

    Parameters:
        noisy_metric (float): Metric value of noisy signal
        denoised_metric (float): Metric value of denoised signal
        higher_is_better (bool): Whether higher metric value is better

    Returns:
        float: Improvement value
    """
    if higher_is_better:
        return denoised_metric - noisy_metric
    else:
        return noisy_metric - denoised_metric


def calculate_percent_improvement(noisy_metric, denoised_metric, higher_is_better=True):
    """
    Calculate percentage improvement of denoising

    Parameters:
        noisy_metric (float): Metric value of noisy signal
        denoised_metric (float): Metric value of denoised signal
        higher_is_better (bool): Whether higher metric value is better

    Returns:
        float: Percentage improvement
    """
    if higher_is_better:
        if noisy_metric == 0:
            return float('inf') if denoised_metric > 0 else 0.0
        return (denoised_metric - noisy_metric) / abs(noisy_metric) * 100
    else:
        if noisy_metric == 0:
            return float('inf') if denoised_metric < 0 else 0.0
        return (noisy_metric - denoised_metric) / abs(noisy_metric) * 100


def paired_t_test(method1_results, method2_results, alpha=0.05):
    """
    Perform paired t-test to compare if performance difference between two methods is significant

    Parameters:
        method1_results (list or numpy.ndarray): Performance metric results for method 1
        method2_results (list or numpy.ndarray): Performance metric results for method 2
        alpha (float): Significance level

    Returns:
        dict: Contains t-statistic, p-value, and boolean indicating statistical significance
    """
    # Ensure inputs are numpy arrays
    if isinstance(method1_results, list):
        method1_results = np.array(method1_results)
    if isinstance(method2_results, list):
        method2_results = np.array(method2_results)

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(method1_results, method2_results)

    # Check if results have statistical significance
    is_significant = p_value < alpha

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'better_method': 'method1' if (t_stat > 0 and is_significant) else
        'method2' if (t_stat < 0 and is_significant) else
        'no significant difference'
    }


def evaluate_denoising(clean_data, noisy_data, denoised_data):
    """
    Comprehensive evaluation of denoising performance

    Parameters:
        clean_data (torch.Tensor or numpy.ndarray): Original clean signal
        noisy_data (torch.Tensor or numpy.ndarray): Noisy signal
        denoised_data (torch.Tensor or numpy.ndarray): Denoised signal

    Returns:
        dict: Dictionary containing various performance metrics
    """
    # Ensure consistent data format
    if isinstance(clean_data, torch.Tensor):
        clean_np = clean_data.cpu().numpy()
        if clean_np.ndim > 1 and clean_np.shape[0] == 1:
            clean_np = clean_np[0]
    else:
        clean_np = clean_data
        if isinstance(clean_data, list):
            clean_np = np.array(clean_data)

    if isinstance(noisy_data, torch.Tensor):
        noisy_np = noisy_data.cpu().numpy()
        if noisy_np.ndim > 1 and noisy_np.shape[0] == 1:
            noisy_np = noisy_np[0]
    else:
        noisy_np = noisy_data
        if isinstance(noisy_data, list):
            noisy_np = np.array(noisy_data)

    if isinstance(denoised_data, torch.Tensor):
        denoised_np = denoised_data.cpu().numpy()
        if denoised_np.ndim > 1 and denoised_np.shape[0] == 1:
            denoised_np = denoised_np[0]
    else:
        denoised_np = denoised_data
        if isinstance(denoised_data, list):
            denoised_np = np.array(denoised_data)

    # Calculate various metrics
    noisy_psnr = calculate_psnr(clean_np, noisy_np)
    denoised_psnr = calculate_psnr(clean_np, denoised_np)
    psnr_improvement = calculate_improvement(noisy_psnr, denoised_psnr)

    noisy_ssim = calculate_ssim(clean_np, noisy_np)
    denoised_ssim = calculate_ssim(clean_np, denoised_np)
    ssim_improvement = calculate_improvement(noisy_ssim, denoised_ssim)

    noisy_mse = calculate_mse(clean_np, noisy_np)
    denoised_mse = calculate_mse(clean_np, denoised_np)
    mse_improvement = calculate_improvement(noisy_mse, denoised_mse, higher_is_better=False)

    # Return all metrics
    return {
        'noisy': {
            'psnr': noisy_psnr,
            'ssim': noisy_ssim,
            'mse': noisy_mse
        },
        'denoised': {
            'psnr': denoised_psnr,
            'ssim': denoised_ssim,
            'mse': denoised_mse
        },
        'improvement': {
            'psnr': psnr_improvement,
            'ssim': ssim_improvement,
            'mse': mse_improvement,
            'psnr_percent': calculate_percent_improvement(noisy_psnr, denoised_psnr),
            'ssim_percent': calculate_percent_improvement(noisy_ssim, denoised_ssim),
            'mse_percent': calculate_percent_improvement(noisy_mse, denoised_mse, higher_is_better=False)
        }
    }


def aggregate_metrics(metrics_list):
    """
    Aggregate evaluation metrics from multiple samples

    Parameters:
        metrics_list (list): List containing evaluation metrics from multiple samples

    Returns:
        dict: Dictionary containing average metrics and standard deviations
    """
    if not metrics_list:
        return {}

    # Initialize result dictionary
    aggregated = {
        'noisy': {'psnr': [], 'ssim': [], 'mse': []},
        'denoised': {'psnr': [], 'ssim': [], 'mse': []},
        'improvement': {'psnr': [], 'ssim': [], 'mse': [],
                        'psnr_percent': [], 'ssim_percent': [], 'mse_percent': []}
    }

    # Collect metrics from all samples
    for metrics in metrics_list:
        for category in aggregated:
            for metric in aggregated[category]:
                if category in metrics and metric in metrics[category]:
                    aggregated[category][metric].append(metrics[category][metric])

    # Calculate averages and standard deviations
    result = {
        'avg': {category: {} for category in aggregated},
        'std': {category: {} for category in aggregated},
        'min': {category: {} for category in aggregated},
        'max': {category: {} for category in aggregated},
        'median': {category: {} for category in aggregated},
        'count': len(metrics_list)
    }

    for category in aggregated:
        for metric in aggregated[category]:
            values = aggregated[category][metric]
            if values:
                values_array = np.array(values)
                result['avg'][category][metric] = float(np.mean(values_array))
                result['std'][category][metric] = float(np.std(values_array))
                result['min'][category][metric] = float(np.min(values_array))
                result['max'][category][metric] = float(np.max(values_array))
                result['median'][category][metric] = float(np.median(values_array))

    return result