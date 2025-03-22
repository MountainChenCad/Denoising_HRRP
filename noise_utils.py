# noise_utils.py
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def add_noise(hrrp_data, noise_level=0.1):
    """
    向HRRP数据添加指定强度的高斯噪声

    参数:
        hrrp_data (torch.Tensor): 干净的HRRP数据
        noise_level (float): 高斯噪声的标准差

    返回:
        torch.Tensor: 带噪声的HRRP数据
    """
    noise = torch.randn_like(hrrp_data) * noise_level
    noisy_data = hrrp_data + noise
    # 确保数据保持在有效范围 [0, 1] 内
    noisy_data = torch.clamp(noisy_data, 0, 1)
    return noisy_data


def add_noise_for_psnr(signal, target_psnr_db):
    """
    添加高斯噪声以达到特定的PSNR(峰值信噪比)值

    参数:
        signal (torch.Tensor): 原始干净信号
        target_psnr_db (float): 期望的PSNR值(dB)

    返回:
        torch.Tensor: 添加噪声后的信号
    """
    # 计算信号功率
    signal_power = torch.mean(signal ** 2)

    # 根据期望的PSNR计算噪声功率
    noise_power = signal_power / (10 ** (target_psnr_db / 10))

    # 生成高斯噪声
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)

    # 将噪声添加到信号中
    noisy_signal = signal + noise

    # 确保数据保持在有效范围 [0, 1] 内
    noisy_signal = torch.clamp(noisy_signal, 0, 1)

    return noisy_signal


def add_noise_for_exact_psnr(signal, target_psnr_db, max_iterations=5, tolerance=0.1):
    """
    添加高斯噪声以精确达到特定的PSNR(峰值信噪比)值

    通过迭代调整噪声强度以确保达到目标PSNR值

    参数:
        signal (torch.Tensor): 原始干净信号
        target_psnr_db (float): 目标PSNR(dB)
        max_iterations (int): 最大迭代调整次数
        tolerance (float): 可接受的PSNR误差范围(dB)

    返回:
        torch.Tensor: 添加噪声后的信号
        float: 实际测得的PSNR
    """
    # 初始估计
    signal_power = torch.mean(signal ** 2)
    noise_power = signal_power / (10 ** (target_psnr_db / 10))

    best_noisy_signal = None
    best_psnr_diff = float('inf')

    for iteration in range(max_iterations):
        # 生成高斯噪声
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)

        # 添加噪声并裁剪
        noisy_signal = signal + noise
        noisy_signal = torch.clamp(noisy_signal, 0, 1)

        # 计算实际PSNR
        actual_psnr = calculate_psnr(signal, noisy_signal)
        psnr_diff = abs(actual_psnr - target_psnr_db)

        # 保存最佳结果
        if psnr_diff < best_psnr_diff:
            best_noisy_signal = noisy_signal.clone()
            best_psnr_diff = psnr_diff

        # 如果足够接近则退出
        if psnr_diff <= tolerance:
            return noisy_signal, actual_psnr

        # 根据当前结果调整噪声功率
        adjustment_factor = 10 ** ((actual_psnr - target_psnr_db) / 20)
        noise_power = noise_power * adjustment_factor

    # 返回最接近目标PSNR的结果
    actual_psnr = calculate_psnr(signal, best_noisy_signal)
    return best_noisy_signal, actual_psnr


def calculate_psnr(original, processed):
    """
    计算PSNR (峰值信噪比)

    参数:
        original (torch.Tensor): 原始干净信号
        processed (torch.Tensor): 噪声信号或恢复信号

    返回:
        float: PSNR值(dB)
    """
    mse = torch.mean((original - processed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(x, y):
    """
    计算两个1D信号之间的SSIM (结构相似性)

    参数:
        x (numpy.ndarray): 第一个信号 (1D)
        y (numpy.ndarray): 第二个信号 (1D)

    返回:
        float: SSIM值在0到1之间 (越高越好)
    """
    # SSIM要求输入为非负
    return ssim(x, y, data_range=1.0)


def psnr_to_noise_level(signal, target_psnr_db):
    """
    将目标PSNR转换为相应的噪声级别(标准差)

    参数:
        signal (torch.Tensor): 原始干净信号
        target_psnr_db (float): 目标PSNR值(dB)

    返回:
        float: 对应的噪声级别(标准差)
    """
    signal_power = torch.mean(signal ** 2)
    noise_power = signal_power / (10 ** (target_psnr_db / 10))
    noise_level = torch.sqrt(noise_power).item()
    return noise_level