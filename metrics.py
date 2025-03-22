# metrics.py - 评估指标计算
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import stats


def calculate_psnr(original, processed):
    """
    计算峰值信噪比 (PSNR)

    参数:
        original (torch.Tensor): 原始干净信号
        processed (torch.Tensor): 处理后的信号(噪声或恢复)

    返回:
        float: PSNR值(dB)
    """
    if isinstance(original, np.ndarray):
        original = torch.from_numpy(original)
    if isinstance(processed, np.ndarray):
        processed = torch.from_numpy(processed)

    # 确保输入张量是浮点型
    original = original.float()
    processed = processed.float()

    # 计算MSE
    mse = torch.mean((original - processed) ** 2)
    if mse == 0:
        return 100.0

    # 假设信号幅度范围为[0, 1]
    max_val = 1.0
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))

    return psnr.item()


def calculate_ssim(x, y):
    """
    计算结构相似性指数 (SSIM)

    参数:
        x (numpy.ndarray): 第一个信号
        y (numpy.ndarray): 第二个信号

    返回:
        float: SSIM值 [0,1] 越高越好
    """
    # 确保输入是numpy数组
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # 如果是批量数据，取第一个样本
    if x.ndim > 1 and x.shape[0] == 1:
        x = x[0]
    if y.ndim > 1 and y.shape[0] == 1:
        y = y[0]

    # SSIM要求输入为非负
    return ssim(x, y, data_range=1.0)


def calculate_mse(original, processed):
    """
    计算均方误差 (MSE)

    参数:
        original (torch.Tensor or numpy.ndarray): 原始干净信号
        processed (torch.Tensor or numpy.ndarray): 处理后的信号

    返回:
        float: MSE值
    """
    if isinstance(original, np.ndarray):
        original = torch.from_numpy(original).float()
    if isinstance(processed, np.ndarray):
        processed = torch.from_numpy(processed).float()

    # 确保输入张量是浮点型
    original = original.float()
    processed = processed.float()

    # 计算MSE
    mse = torch.mean((original - processed) ** 2)

    return mse.item()


def calculate_improvement(noisy_metric, denoised_metric, higher_is_better=True):
    """
    计算去噪改进量

    参数:
        noisy_metric (float): 噪声信号的指标值
        denoised_metric (float): 去噪信号的指标值
        higher_is_better (bool): 指标值是否越高越好

    返回:
        float: 改进量
    """
    if higher_is_better:
        return denoised_metric - noisy_metric
    else:
        return noisy_metric - denoised_metric


def calculate_percent_improvement(noisy_metric, denoised_metric, higher_is_better=True):
    """
    计算去噪百分比改进

    参数:
        noisy_metric (float): 噪声信号的指标值
        denoised_metric (float): 去噪信号的指标值
        higher_is_better (bool): 指标值是否越高越好

    返回:
        float: 百分比改进
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
    执行配对t检验以比较两种方法的性能差异是否显著

    参数:
        method1_results (list or numpy.ndarray): 方法1的性能指标结果
        method2_results (list or numpy.ndarray): 方法2的性能指标结果
        alpha (float): 显著性水平

    返回:
        dict: 包含t统计量、p值和是否具有统计显著性差异的布尔值
    """
    # 确保输入是numpy数组
    if isinstance(method1_results, list):
        method1_results = np.array(method1_results)
    if isinstance(method2_results, list):
        method2_results = np.array(method2_results)

    # 执行配对t检验
    t_stat, p_value = stats.ttest_rel(method1_results, method2_results)

    # 检查结果是否具有统计显著性
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
    全面评估去噪性能

    参数:
        clean_data (torch.Tensor or numpy.ndarray): 原始干净信号
        noisy_data (torch.Tensor or numpy.ndarray): 噪声信号
        denoised_data (torch.Tensor or numpy.ndarray): 去噪信号

    返回:
        dict: 包含各种性能指标的字典
    """
    # 确保数据格式一致
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

    # 计算各种指标
    noisy_psnr = calculate_psnr(clean_np, noisy_np)
    denoised_psnr = calculate_psnr(clean_np, denoised_np)
    psnr_improvement = calculate_improvement(noisy_psnr, denoised_psnr)

    noisy_ssim = calculate_ssim(clean_np, noisy_np)
    denoised_ssim = calculate_ssim(clean_np, denoised_np)
    ssim_improvement = calculate_improvement(noisy_ssim, denoised_ssim)

    noisy_mse = calculate_mse(clean_np, noisy_np)
    denoised_mse = calculate_mse(clean_np, denoised_np)
    mse_improvement = calculate_improvement(noisy_mse, denoised_mse, higher_is_better=False)

    # 返回所有指标
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
    聚合多个样本的评估指标

    参数:
        metrics_list (list): 包含多个样本评估指标的列表

    返回:
        dict: 包含平均指标和标准差的字典
    """
    if not metrics_list:
        return {}

    # 初始化结果字典
    aggregated = {
        'noisy': {'psnr': [], 'ssim': [], 'mse': []},
        'denoised': {'psnr': [], 'ssim': [], 'mse': []},
        'improvement': {'psnr': [], 'ssim': [], 'mse': [],
                        'psnr_percent': [], 'ssim_percent': [], 'mse_percent': []}
    }

    # 收集所有样本的指标
    for metrics in metrics_list:
        for category in aggregated:
            for metric in aggregated[category]:
                if category in metrics and metric in metrics[category]:
                    aggregated[category][metric].append(metrics[category][metric])

    # 计算平均值和标准差
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