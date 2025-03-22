# test_all.py - 统一测试入口
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import time
from datetime import datetime
import random
import json

# 导入模型定义
from models import TargetRadialLengthModule, TargetIdentityModule
from cgan_models import Generator
from cae_models import ConvAutoEncoder
from ae_models import AutoEncoder
from hrrp_dataset import HRRPDataset
from torch.utils.data import DataLoader
from noise_utils import add_noise_for_exact_psnr, calculate_psnr, calculate_ssim

# 设置matplotlib参数以获得高质量可视化
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# 定义科学绘图的配色方案
COLORS = {
    'noisy': '#7F7F7F',  # 灰色
    'cgan': '#1F77B4',  # 蓝色
    'cae': '#2CA02C',  # 绿色
    'ae': '#D62728',  # 红色
    'clean': '#000000'  # 黑色
}


def setup_seed(seed):
    """设置随机种子以确保可重现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def test_cgan(args, device, psnr_level):
    """
    测试CGAN在特定PSNR级别下的去噪性能

    参数:
        args: 命令行参数
        device: 计算设备(CPU/GPU)
        psnr_level: 测试的PSNR级别(dB)

    返回:
        dict: 包含测试结果的字典
    """
    print(f"\n{'-' * 20} 测试CGAN (PSNR={psnr_level}dB) {'-' * 20}")

    # 为当前PSNR级别创建输出目录
    output_dir = os.path.join(args.output_dir, f"cgan_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # 记录测试结果
    results = {
        "model": "CGAN",
        "psnr_level": psnr_level,
        "metrics": {
            "mse": [],
            "psnr": [],
            "ssim": [],
            "noisy_psnr": []
        },
        "averages": {}
    }

    # 加载特征提取器
    G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)
    G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                               num_classes=args.num_classes).to(device)

    # 加载其权重
    G_D.load_state_dict(torch.load(os.path.join(args.cgan_dir, f"psnr_{psnr_level}dB", 'G_D_final.pth')))
    G_I.load_state_dict(torch.load(os.path.join(args.cgan_dir, f"psnr_{psnr_level}dB", 'G_I_final.pth')))

    # 设置为评估模式
    G_D.eval()
    G_I.eval()

    # 加载生成器
    generator = Generator(input_dim=args.input_dim,
                          condition_dim=args.feature_dim * 2,
                          hidden_dim=args.hidden_dim).to(device)

    # 加载生成器权重
    generator.load_state_dict(torch.load(os.path.join(args.cgan_dir, f"psnr_{psnr_level}dB", 'generator_final.pth')))
    generator.eval()

    # 加载测试数据集
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 定义用于评估的损失函数
    mse_loss = nn.MSELoss()

    # 对测试样本进行去噪
    with torch.no_grad():
        for i, (clean_data, radial_length, identity_label) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # 将数据移至设备
            clean_data = clean_data.float().to(device)

            # 创建噪声数据，尝试精确匹配目标PSNR
            noisy_data, actual_psnr = add_noise_for_exact_psnr(clean_data, psnr_level)

            # 提取特征并创建条件
            f_D, _ = G_D(clean_data)
            f_I, _ = G_I(clean_data)
            condition = torch.cat([f_D, f_I], dim=1)

            # 生成去噪样本
            denoised_data = generator(noisy_data, condition)

            # 计算指标
            mse = mse_loss(denoised_data, clean_data).item()
            psnr = calculate_psnr(clean_data, denoised_data)

            # 计算SSIM (需要转换为numpy数组)
            clean_np = clean_data.cpu().numpy()[0]
            denoised_np = denoised_data.cpu().numpy()[0]
            ssim = calculate_ssim(clean_np, denoised_np)

            # 保存指标
            results["metrics"]["mse"].append(mse)
            results["metrics"]["psnr"].append(psnr)
            results["metrics"]["ssim"].append(ssim)
            results["metrics"]["noisy_psnr"].append(actual_psnr)

            # 绘制结果
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(clean_data.cpu().numpy()[0], color=COLORS['clean'])
            plt.title('Clean HRRP')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.subplot(1, 3, 2)
            plt.plot(noisy_data.cpu().numpy()[0], color=COLORS['noisy'])
            plt.title(f'Noisy HRRP (PSNR: {actual_psnr:.2f}dB)')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.subplot(1, 3, 3)
            plt.plot(denoised_data.cpu().numpy()[0], color=COLORS['cgan'])
            plt.title(f'CGAN Denoised (PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i + 1}_denoising.png'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"样本 {i + 1}:")
            print(f"  噪声 PSNR: {actual_psnr:.2f}dB")
            print(f"  去噪 PSNR: {psnr:.2f}dB")
            print(f"  去噪 SSIM: {ssim:.4f}")
            print(f"  去噪 MSE: {mse:.6f}")
            print(f"  改进: {psnr - actual_psnr:.2f}dB")
            print()

    # 计算平均指标
    avg_mse = np.mean(results["metrics"]["mse"])
    avg_psnr = np.mean(results["metrics"]["psnr"])
    avg_ssim = np.mean(results["metrics"]["ssim"])
    avg_noisy_psnr = np.mean(results["metrics"]["noisy_psnr"])
    avg_improvement = avg_psnr - avg_noisy_psnr

    # 保存平均指标
    results["averages"]["mse"] = float(avg_mse)
    results["averages"]["psnr"] = float(avg_psnr)
    results["averages"]["ssim"] = float(avg_ssim)
    results["averages"]["noisy_psnr"] = float(avg_noisy_psnr)
    results["averages"]["improvement"] = float(avg_improvement)

    # 输出平均指标
    print(f"\nCGAN平均指标 (PSNR={psnr_level}dB):")
    print(f"  噪声 PSNR: {avg_noisy_psnr:.2f}dB")
    print(f"  去噪 PSNR: {avg_psnr:.2f}dB")
    print(f"  去噪 SSIM: {avg_ssim:.4f}")
    print(f"  去噪 MSE: {avg_mse:.6f}")
    print(f"  平均改进: {avg_improvement:.2f}dB")

    # 保存结果到JSON文件
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 创建结果摘要文本文件
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"CGAN测试结果 (PSNR={psnr_level}dB)\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"测试样本数: {min(args.num_samples, len(test_loader))}\n\n")
        f.write(f"平均指标:\n")
        f.write(f"  噪声 PSNR: {avg_noisy_psnr:.2f}dB\n")
        f.write(f"  去噪 PSNR: {avg_psnr:.2f}dB\n")
        f.write(f"  去噪 SSIM: {avg_ssim:.4f}\n")
        f.write(f"  去噪 MSE: {avg_mse:.6f}\n")
        f.write(f"  平均改进: {avg_improvement:.2f}dB\n")

    return results


def test_cae(args, device, psnr_level):
    """
    测试CAE在特定PSNR级别下的去噪性能

    参数:
        args: 命令行参数
        device: 计算设备(CPU/GPU)
        psnr_level: 测试的PSNR级别(dB)

    返回:
        dict: 包含测试结果的字典
    """
    print(f"\n{'-' * 20} 测试CAE (PSNR={psnr_level}dB) {'-' * 20}")

    # 为当前PSNR级别创建输出目录
    output_dir = os.path.join(args.output_dir, f"cae_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # 记录测试结果
    results = {
        "model": "CAE",
        "psnr_level": psnr_level,
        "metrics": {
            "mse": [],
            "psnr": [],
            "ssim": [],
            "noisy_psnr": []
        },
        "averages": {}
    }

    # 加载CAE模型
    model = ConvAutoEncoder(input_dim=args.input_dim,
                            latent_dim=args.latent_dim,
                            hidden_dim=args.hidden_dim).to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(os.path.join(args.cae_dir, f"psnr_{psnr_level}dB", 'cae_model_final.pth')))
    model.eval()

    # 加载测试数据集
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 定义用于评估的损失函数
    mse_loss = nn.MSELoss()

    # 对测试样本进行去噪
    with torch.no_grad():
        for i, (clean_data, _, _) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # 将数据移至设备
            clean_data = clean_data.float().to(device)

            # 创建噪声数据
            noisy_data, actual_psnr = add_noise_for_exact_psnr(clean_data, psnr_level)

            # 生成去噪样本
            denoised_data, _ = model(noisy_data)

            # 计算指标
            mse = mse_loss(denoised_data, clean_data).item()
            psnr = calculate_psnr(clean_data, denoised_data)

            # 计算SSIM (需要转换为numpy数组)
            clean_np = clean_data.cpu().numpy()[0]
            denoised_np = denoised_data.cpu().numpy()[0]
            ssim = calculate_ssim(clean_np, denoised_np)

            # 保存指标
            results["metrics"]["mse"].append(mse)
            results["metrics"]["psnr"].append(psnr)
            results["metrics"]["ssim"].append(ssim)
            results["metrics"]["noisy_psnr"].append(actual_psnr)

            # 绘制结果
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(clean_data.cpu().numpy()[0], color=COLORS['clean'])
            plt.title('Clean HRRP')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.subplot(1, 3, 2)
            plt.plot(noisy_data.cpu().numpy()[0], color=COLORS['noisy'])
            plt.title(f'Noisy HRRP (PSNR: {actual_psnr:.2f}dB)')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.subplot(1, 3, 3)
            plt.plot(denoised_data.cpu().numpy()[0], color=COLORS['cae'])
            plt.title(f'CAE Denoised (PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i + 1}_denoising.png'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"样本 {i + 1}:")
            print(f"  噪声 PSNR: {actual_psnr:.2f}dB")
            print(f"  去噪 PSNR: {psnr:.2f}dB")
            print(f"  去噪 SSIM: {ssim:.4f}")
            print(f"  去噪 MSE: {mse:.6f}")
            print(f"  改进: {psnr - actual_psnr:.2f}dB")
            print()

    # 计算平均指标
    avg_mse = np.mean(results["metrics"]["mse"])
    avg_psnr = np.mean(results["metrics"]["psnr"])
    avg_ssim = np.mean(results["metrics"]["ssim"])
    avg_noisy_psnr = np.mean(results["metrics"]["noisy_psnr"])
    avg_improvement = avg_psnr - avg_noisy_psnr

    # 保存平均指标
    results["averages"]["mse"] = float(avg_mse)
    results["averages"]["psnr"] = float(avg_psnr)
    results["averages"]["ssim"] = float(avg_ssim)
    results["averages"]["noisy_psnr"] = float(avg_noisy_psnr)
    results["averages"]["improvement"] = float(avg_improvement)

    # 输出平均指标
    print(f"\nCAE平均指标 (PSNR={psnr_level}dB):")
    print(f"  噪声 PSNR: {avg_noisy_psnr:.2f}dB")
    print(f"  去噪 PSNR: {avg_psnr:.2f}dB")
    print(f"  去噪 SSIM: {avg_ssim:.4f}")
    print(f"  去噪 MSE: {avg_mse:.6f}")
    print(f"  平均改进: {avg_improvement:.2f}dB")

    # 保存结果到JSON文件
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 创建结果摘要文本文件
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"CAE测试结果 (PSNR={psnr_level}dB)\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"测试样本数: {min(args.num_samples, len(test_loader))}\n\n")
        f.write(f"平均指标:\n")
        f.write(f"  噪声 PSNR: {avg_noisy_psnr:.2f}dB\n")
        f.write(f"  去噪 PSNR: {avg_psnr:.2f}dB\n")
        f.write(f"  去噪 SSIM: {avg_ssim:.4f}\n")
        f.write(f"  去噪 MSE: {avg_mse:.6f}\n")
        f.write(f"  平均改进: {avg_improvement:.2f}dB\n")

    return results


def test_ae(args, device, psnr_level):
    """
    测试AE在特定PSNR级别下的去噪性能

    参数:
        args: 命令行参数
        device: 计算设备(CPU/GPU)
        psnr_level: 测试的PSNR级别(dB)

    返回:
        dict: 包含测试结果的字典
    """
    print(f"\n{'-' * 20} 测试AE (PSNR={psnr_level}dB) {'-' * 20}")

    # 为当前PSNR级别创建输出目录
    output_dir = os.path.join(args.output_dir, f"ae_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # 记录测试结果
    results = {
        "model": "AE",
        "psnr_level": psnr_level,
        "metrics": {
            "mse": [],
            "psnr": [],
            "ssim": [],
            "noisy_psnr": []
        },
        "averages": {}
    }

    # 加载AE模型
    model = AutoEncoder(input_dim=args.input_dim,
                        latent_dim=args.latent_dim,
                        hidden_dim=args.ae_hidden_dim).to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(os.path.join(args.ae_dir, f"psnr_{psnr_level}dB", 'ae_model_final.pth')))
    model.eval()

    # 加载测试数据集
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 定义用于评估的损失函数
    mse_loss = nn.MSELoss()

    # 对测试样本进行去噪
    with torch.no_grad():
        for i, (clean_data, _, _) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # 将数据移至设备
            clean_data = clean_data.float().to(device)

            # 创建噪声数据
            noisy_data, actual_psnr = add_noise_for_exact_psnr(clean_data, psnr_level)

            # 生成去噪样本
            denoised_data, _ = model(noisy_data)

            # 计算指标
            mse = mse_loss(denoised_data, clean_data).item()
            psnr = calculate_psnr(clean_data, denoised_data)

            # 计算SSIM (需要转换为numpy数组)
            clean_np = clean_data.cpu().numpy()[0]
            denoised_np = denoised_data.cpu().numpy()[0]
            ssim = calculate_ssim(clean_np, denoised_np)

            # 保存指标
            results["metrics"]["mse"].append(mse)
            results["metrics"]["psnr"].append(psnr)
            results["metrics"]["ssim"].append(ssim)
            results["metrics"]["noisy_psnr"].append(actual_psnr)

            # 绘制结果
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(clean_data.cpu().numpy()[0], color=COLORS['clean'])
            plt.title('Clean HRRP')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.subplot(1, 3, 2)
            plt.plot(noisy_data.cpu().numpy()[0], color=COLORS['noisy'])
            plt.title(f'Noisy HRRP (PSNR: {actual_psnr:.2f}dB)')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.subplot(1, 3, 3)
            plt.plot(denoised_data.cpu().numpy()[0], color=COLORS['ae'])
            plt.title(f'AE Denoised (PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i + 1}_denoising.png'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"样本 {i + 1}:")
            print(f"  噪声 PSNR: {actual_psnr:.2f}dB")
            print(f"  去噪 PSNR: {psnr:.2f}dB")
            print(f"  去噪 SSIM: {ssim:.4f}")
            print(f"  去噪 MSE: {mse:.6f}")
            print(f"  改进: {psnr - actual_psnr:.2f}dB")
            print()

    # 计算平均指标
    avg_mse = np.mean(results["metrics"]["mse"])
    avg_psnr = np.mean(results["metrics"]["psnr"])
    avg_ssim = np.mean(results["metrics"]["ssim"])
    avg_noisy_psnr = np.mean(results["metrics"]["noisy_psnr"])
    avg_improvement = avg_psnr - avg_noisy_psnr

    # 保存平均指标
    results["averages"]["mse"] = float(avg_mse)
    results["averages"]["psnr"] = float(avg_psnr)
    results["averages"]["ssim"] = float(avg_ssim)
    results["averages"]["noisy_psnr"] = float(avg_noisy_psnr)
    results["averages"]["improvement"] = float(avg_improvement)

    # 输出平均指标
    print(f"\nAE平均指标 (PSNR={psnr_level}dB):")
    print(f"  噪声 PSNR: {avg_noisy_psnr:.2f}dB")
    print(f"  去噪 PSNR: {avg_psnr:.2f}dB")
    print(f"  去噪 SSIM: {avg_ssim:.4f}")
    print(f"  去噪 MSE: {avg_mse:.6f}")
    print(f"  平均改进: {avg_improvement:.2f}dB")

    # 保存结果到JSON文件
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 创建结果摘要文本文件
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"AE测试结果 (PSNR={psnr_level}dB)\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"测试样本数: {min(args.num_samples, len(test_loader))}\n\n")
        f.write(f"平均指标:\n")
        f.write(f"  噪声 PSNR: {avg_noisy_psnr:.2f}dB\n")
        f.write(f"  去噪 PSNR: {avg_psnr:.2f}dB\n")
        f.write(f"  去噪 SSIM: {avg_ssim:.4f}\n")
        f.write(f"  去噪 MSE: {avg_mse:.6f}\n")
        f.write(f"  平均改进: {avg_improvement:.2f}dB\n")

    return results


def compare_models(args, device, psnr_level, model_results):
    """
    比较所有模型在特定PSNR级别下的性能并创建可视化比较

    参数:
        args: 命令行参数
        device: 计算设备(CPU/GPU)
        psnr_level: 测试的PSNR级别(dB)
        model_results: 包含各模型测试结果的字典
    """
    print(f"\n{'-' * 20} 模型比较 (PSNR={psnr_level}dB) {'-' * 20}")

    # 为当前PSNR级别创建比较结果目录
    output_dir = os.path.join(args.output_dir, f"comparison_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # 提取各模型的性能指标
    models = list(model_results.keys())

    # 准备绘图数据
    metrics = {
        "psnr": [],
        "ssim": [],
        "mse": [],
        "improvement": []
    }

    for model in models:
        metrics["psnr"].append(model_results[model]["averages"]["psnr"])
        metrics["ssim"].append(model_results[model]["averages"]["ssim"])
        metrics["mse"].append(model_results[model]["averages"]["mse"])
        metrics["improvement"].append(model_results[model]["averages"]["improvement"])

    # 添加噪声PSNR作为参考 (从任意模型获取，应该相同)
    noisy_psnr = model_results[models[0]]["averages"]["noisy_psnr"]

    # 绘制模型比较条形图
    plt.figure(figsize=(15, 12))

    # PSNR比较图
    plt.subplot(3, 1, 1)
    x = np.arange(len(models) + 1)  # +1 for noisy signal
    values = [noisy_psnr] + metrics["psnr"]
    colors = [COLORS['noisy']] + [COLORS[model.lower()] for model in models]

    bars = plt.bar(x, values, color=colors, width=0.6, edgecolor='k', linewidth=1)
    plt.title(f'PSNR Comparison at {psnr_level}dB Noise Level', fontsize=14)
    plt.xlabel('Methods')
    plt.ylabel('PSNR (dB)')
    plt.xticks(x, ['Noisy'] + models)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 在条形上方添加PSNR值和改善量
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == 0:  # 噪声信号，没有改善
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{height:.2f}dB', ha='center', va='bottom')
        else:  # 去噪方法，显示改善量
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{height:.2f}dB\n(+{metrics["improvement"][i - 1]:.2f}dB)',
                     ha='center', va='bottom')

    # MSE比较图
    plt.subplot(3, 1, 2)
    bars = plt.bar(np.arange(len(models)), metrics["mse"],
                   color=[COLORS[model.lower()] for model in models],
                   width=0.6, edgecolor='k', linewidth=1)
    plt.title('MSE Comparison', fontsize=14)
    plt.xlabel('Methods')
    plt.ylabel('Mean Squared Error')
    plt.xticks(np.arange(len(models)), models)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 添加MSE值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.0005,
                 f'{height:.6f}', ha='center', va='bottom')

    # SSIM比较图
    plt.subplot(3, 1, 3)
    bars = plt.bar(np.arange(len(models)), metrics["ssim"],
                   color=[COLORS[model.lower()] for model in models],
                   width=0.6, edgecolor='k', linewidth=1)
    plt.title('SSIM Comparison', fontsize=14)
    plt.xlabel('Methods')
    plt.ylabel('Structural Similarity Index')
    plt.xticks(np.arange(len(models)), models)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 调整SSIM轴的范围以突出差异
    ssim_min = min(metrics["ssim"]) * 0.95
    plt.ylim([ssim_min, 1.0])

    # 添加SSIM值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 创建比较结果文本文件
    with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
        f.write(f"模型比较结果 (PSNR={psnr_level}dB)\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"噪声信号 PSNR: {noisy_psnr:.2f}dB\n\n")

        f.write("各模型性能指标:\n")
        for i, model in enumerate(models):
            f.write(f"  {model}:\n")
            f.write(f"    PSNR: {metrics['psnr'][i]:.2f}dB (提升: +{metrics['improvement'][i]:.2f}dB)\n")
            f.write(f"    SSIM: {metrics['ssim'][i]:.4f}\n")
            f.write(f"    MSE: {metrics['mse'][i]:.6f}\n\n")

        # 找出最佳模型
        best_psnr_idx = np.argmax(metrics["psnr"])
        best_ssim_idx = np.argmax(metrics["ssim"])
        best_mse_idx = np.argmin(metrics["mse"])

        f.write("最佳性能模型:\n")
        f.write(f"  按PSNR: {models[best_psnr_idx]} ({metrics['psnr'][best_psnr_idx]:.2f}dB)\n")
        f.write(f"  按SSIM: {models[best_ssim_idx]} ({metrics['ssim'][best_ssim_idx]:.4f})\n")
        f.write(f"  按MSE: {models[best_mse_idx]} ({metrics['mse'][best_mse_idx]:.6f})\n")

    print(f"\n模型比较结果已保存至: {output_dir}")

    # 返回最佳模型
    return {
        "best_psnr_model": models[best_psnr_idx],
        "best_ssim_model": models[best_ssim_idx],
        "best_mse_model": models[best_mse_idx]
    }


def create_psnr_summary(args, all_results):
    """
    创建跨所有PSNR级别的性能总结和可视化

    参数:
        args: 命令行参数
        all_results: 包含所有PSNR级别和所有模型结果的字典
    """
    print(f"\n{'-' * 20} 创建PSNR总结 {'-' * 20}")

    # 创建总结目录
    summary_dir = os.path.join(args.output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # 获取所有PSNR级别和模型
    psnr_levels = sorted(all_results.keys())
    models = sorted(all_results[psnr_levels[0]].keys())

    # 准备数据
    data = {model: {
        "psnr": [],
        "ssim": [],
        "improvement": []
    } for model in models}

    # 提取所有PSNR级别的性能指标
    for psnr_level in psnr_levels:
        for model in models:
            data[model]["psnr"].append(all_results[psnr_level][model]["averages"]["psnr"])
            data[model]["ssim"].append(all_results[psnr_level][model]["averages"]["ssim"])
            data[model]["improvement"].append(all_results[psnr_level][model]["averages"]["improvement"])

    # 创建PSNR改善对比图
    plt.figure(figsize=(12, 6))

    width = 0.25
    x = np.arange(len(psnr_levels))

    for i, model in enumerate(models):
        plt.bar(x + (i - 1) * width, data[model]["improvement"], width,
                label=model, color=COLORS[model.lower()], edgecolor='k', linewidth=1)

    plt.xlabel('Input Noise Level (PSNR in dB)')
    plt.ylabel('PSNR Improvement (dB)')
    plt.title('PSNR Improvement vs. Noise Level')
    plt.xticks(x, [f'{level}dB' for level in psnr_levels])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'psnr_improvement_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 创建SSIM对比图
    plt.figure(figsize=(12, 6))

    for i, model in enumerate(models):
        plt.bar(x + (i - 1) * width, data[model]["ssim"], width,
                label=model, color=COLORS[model.lower()], edgecolor='k', linewidth=1)

    plt.xlabel('Input Noise Level (PSNR in dB)')
    plt.ylabel('SSIM')
    plt.title('SSIM vs. Noise Level')
    plt.xticks(x, [f'{level}dB' for level in psnr_levels])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'ssim_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 创建总结报告
    with open(os.path.join(summary_dir, 'overall_summary.txt'), 'w') as f:
        f.write("HRRP去噪方法总体性能评估\n")
        f.write(f"{'=' * 50}\n\n")

        # 写入每个PSNR级别的总结
        for i, psnr_level in enumerate(psnr_levels):
            f.write(f"PSNR = {psnr_level}dB 条件下的结果:\n")

            for model in models:
                f.write(f"  {model}:\n")
                f.write(f"    PSNR: {data[model]['psnr'][i]:.2f}dB\n")
                f.write(f"    SSIM: {data[model]['ssim'][i]:.4f}\n")
                f.write(f"    改善: +{data[model]['improvement'][i]:.2f}dB\n")

            f.write("\n")

        # 计算每个模型在多少个PSNR级别上表现最佳
        best_psnr_counts = {model: 0 for model in models}
        best_ssim_counts = {model: 0 for model in models}

        for psnr_level in psnr_levels:
            # 找出PSNR最佳模型
            psnr_values = [all_results[psnr_level][model]["averages"]["psnr"] for model in models]
            best_psnr_model = models[np.argmax(psnr_values)]
            best_psnr_counts[best_psnr_model] += 1

            # 找出SSIM最佳模型
            ssim_values = [all_results[psnr_level][model]["averages"]["ssim"] for model in models]
            best_ssim_model = models[np.argmax(ssim_values)]
            best_ssim_counts[best_ssim_model] += 1

        f.write("\n总体最佳模型统计:\n")
        f.write(f"  按PSNR计算，各方法获胜次数:\n")
        for model, count in best_psnr_counts.items():
            f.write(f"    {model}: {count}/{len(psnr_levels)} 次\n")

        f.write(f"\n  按SSIM计算，各方法获胜次数:\n")
        for model, count in best_ssim_counts.items():
            f.write(f"    {model}: {count}/{len(psnr_levels)} 次\n")

    # 保存所有结果到JSON文件
    with open(os.path.join(summary_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n总体性能评估结果已保存至: {summary_dir}")


def test_sample_visualization(args, device, psnr_level):
    """
    为特定PSNR级别创建所有模型的样本可视化比较

    参数:
        args: 命令行参数
        device: 计算设备(CPU/GPU)
        psnr_level: 测试的PSNR级别(dB)
    """
    print(f"\n{'-' * 20} 创建样本可视化 (PSNR={psnr_level}dB) {'-' * 20}")

    # 为样本可视化创建目录
    samples_dir = os.path.join(args.output_dir, f"samples_psnr_{psnr_level}dB")
    os.makedirs(samples_dir, exist_ok=True)

    # 加载模型
    # 1. CGAN模型
    G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)
    G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                               num_classes=args.num_classes).to(device)
    cgan_generator = Generator(input_dim=args.input_dim, condition_dim=args.feature_dim * 2,
                               hidden_dim=args.hidden_dim).to(device)

    G_D.load_state_dict(torch.load(os.path.join(args.cgan_dir, f"psnr_{psnr_level}dB", 'G_D_final.pth')))
    G_I.load_state_dict(torch.load(os.path.join(args.cgan_dir, f"psnr_{psnr_level}dB", 'G_I_final.pth')))
    cgan_generator.load_state_dict(
        torch.load(os.path.join(args.cgan_dir, f"psnr_{psnr_level}dB", 'generator_final.pth')))

    G_D.eval()
    G_I.eval()
    cgan_generator.eval()

    # 2. CAE模型
    cae_model = ConvAutoEncoder(input_dim=args.input_dim,
                                latent_dim=args.latent_dim,
                                hidden_dim=args.hidden_dim).to(device)
    cae_model.load_state_dict(torch.load(os.path.join(args.cae_dir, f"psnr_{psnr_level}dB", 'cae_model_final.pth')))
    cae_model.eval()

    # 3. AE模型
    ae_model = AutoEncoder(input_dim=args.input_dim,
                           latent_dim=args.latent_dim,
                           hidden_dim=args.ae_hidden_dim).to(device)
    ae_model.load_state_dict(torch.load(os.path.join(args.ae_dir, f"psnr_{psnr_level}dB", 'ae_model_final.pth')))
    ae_model.eval()

    # 加载测试数据集
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 定义损失函数
    mse_loss = nn.MSELoss()

    # 创建样本可视化
    with torch.no_grad():
        for i, (clean_data, radial_length, identity_label) in enumerate(test_loader):
            if i >= args.num_vis_samples:
                break

            # 将数据移至设备
            clean_data = clean_data.float().to(device)

            # 创建噪声数据
            noisy_data, actual_psnr = add_noise_for_exact_psnr(clean_data, psnr_level)

            # 模型推理
            # 1. CGAN
            f_D, _ = G_D(clean_data)
            f_I, _ = G_I(clean_data)
            condition = torch.cat([f_D, f_I], dim=1)
            cgan_denoised = cgan_generator(noisy_data, condition)

            # 2. CAE
            cae_denoised, _ = cae_model(noisy_data)

            # 3. AE
            ae_denoised, _ = ae_model(noisy_data)

            # 计算指标
            cgan_mse = mse_loss(cgan_denoised, clean_data).item()
            cae_mse = mse_loss(cae_denoised, clean_data).item()
            ae_mse = mse_loss(ae_denoised, clean_data).item()

            cgan_psnr = calculate_psnr(clean_data, cgan_denoised)
            cae_psnr = calculate_psnr(clean_data, cae_denoised)
            ae_psnr = calculate_psnr(clean_data, ae_denoised)

            # 转换为numpy数组
            clean_np = clean_data.cpu().numpy()[0]
            noisy_np = noisy_data.cpu().numpy()[0]
            cgan_np = cgan_denoised.cpu().numpy()[0]
            cae_np = cae_denoised.cpu().numpy()[0]
            ae_np = ae_denoised.cpu().numpy()[0]

            # 计算SSIM
            cgan_ssim = calculate_ssim(clean_np, cgan_np)
            cae_ssim = calculate_ssim(clean_np, cae_np)
            ae_ssim = calculate_ssim(clean_np, ae_np)

            # 创建并保存可视化
            plt.figure(figsize=(15, 10))

            # 1. 原始信号
            plt.subplot(2, 3, 1)
            plt.plot(clean_np, color=COLORS['clean'], linewidth=1.5)
            plt.title('Clean HRRP')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            # 2. 噪声信号
            plt.subplot(2, 3, 2)
            plt.plot(noisy_np, color=COLORS['noisy'], linewidth=1.5)
            plt.title(f'Noisy HRRP (PSNR: {actual_psnr:.2f}dB)')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            # 3. CGAN结果
            plt.subplot(2, 3, 3)
            plt.plot(cgan_np, color=COLORS['cgan'], linewidth=1.5)
            plt.title(f'CGAN (PSNR: {cgan_psnr:.2f}dB, SSIM: {cgan_ssim:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            # 4. CAE结果
            plt.subplot(2, 3, 5)
            plt.plot(cae_np, color=COLORS['cae'], linewidth=1.5)
            plt.title(f'CAE (PSNR: {cae_psnr:.2f}dB, SSIM: {cae_ssim:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            # 5. AE结果
            plt.subplot(2, 3, 6)
            plt.plot(ae_np, color=COLORS['ae'], linewidth=1.5)
            plt.title(f'AE (PSNR: {ae_psnr:.2f}dB, SSIM: {ae_ssim:.4f})')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, linestyle='--', alpha=0.3)

            # 添加总标题
            plt.suptitle(f'HRRP Denoising Methods Comparison (PSNR={psnr_level}dB)', fontsize=16)

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # 为suptitle留出空间
            plt.savefig(os.path.join(samples_dir, f'sample_{i + 1}_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 创建叠加比较图
            plt.figure(figsize=(12, 6))

            plt.plot(clean_np, color=COLORS['clean'], linewidth=1.5, label='Clean')
            plt.plot(noisy_np, color=COLORS['noisy'], linewidth=1.0, alpha=0.6, label='Noisy')
            plt.plot(cgan_np, color=COLORS['cgan'], linewidth=1.2, label='CGAN')
            plt.plot(cae_np, color=COLORS['cae'], linewidth=1.2, label='CAE')
            plt.plot(ae_np, color=COLORS['ae'], linewidth=1.2, label='AE')

            plt.title(f'HRRP Denoising Methods Comparison (PSNR={psnr_level}dB)', fontsize=14)
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f'sample_{i + 1}_overlay.png'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"创建样本 {i + 1} 的可视化比较")

    print(f"\n样本可视化已保存至: {samples_dir}")


def main():
    """主函数：解析命令行参数，执行测试流程"""
    parser = argparse.ArgumentParser(description='HRRP去噪模型统一测试脚本')

    # 基本参数
    parser.add_argument('--model', type=str, default='all', choices=['cgan', 'cae', 'ae', 'all'],
                        help='要测试的模型类型: cgan, cae, ae 或 all(测试所有模型)')
    parser.add_argument('--test_dir', type=str, default='datasets/simulated_3/test',
                        help='包含测试数据的目录')
    parser.add_argument('--psnr_levels', type=float, nargs='+', default=[20, 10, 0],
                        help='要测试的PSNR级别列表(dB)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 模型加载参数
    parser.add_argument('--cgan_dir', type=str, default='checkpoints/cgan',
                        help='CGAN模型目录')
    parser.add_argument('--cae_dir', type=str, default='checkpoints/cae',
                        help='CAE模型目录')
    parser.add_argument('--ae_dir', type=str, default='checkpoints/ae',
                        help='AE模型目录')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results',
                        help='保存测试结果的目录')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='测试样本数量')
    parser.add_argument('--num_vis_samples', type=int, default=5,
                        help='创建可视化比较的样本数量')

    # 网络参数
    parser.add_argument('--input_dim', type=int, default=500,
                        help='输入HRRP序列的维度')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='特征提取器输出的维度')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='CGAN和CAE隐藏层的维度')
    parser.add_argument('--ae_hidden_dim', type=int, default=256,
                        help='AE隐藏层的维度')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='CAE和AE的潜在空间维度')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='目标身份类别的数量')

    args = parser.parse_args()

    # 设置随机种子
    setup_seed(args.seed)

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 测试开始时间
    start_time = time.time()

    # 存储所有结果
    all_results = {}

    # 对每个PSNR级别测试指定的模型
    for psnr_level in args.psnr_levels:
        print(f"\n{'=' * 50}")
        print(f"开始PSNR={psnr_level}dB测试")
        print(f"{'=' * 50}")

        # 初始化当前PSNR级别的结果存储
        all_results[psnr_level] = {}
        model_results = {}

        # 测试CGAN (如果选择)
        if args.model in ['cgan', 'all']:
            cgan_results = test_cgan(args, device, psnr_level)
            model_results['CGAN'] = cgan_results
            all_results[psnr_level]['CGAN'] = cgan_results

        # 测试CAE (如果选择)
        if args.model in ['cae', 'all']:
            cae_results = test_cae(args, device, psnr_level)
            model_results['CAE'] = cae_results
            all_results[psnr_level]['CAE'] = cae_results

        # 测试AE (如果选择)
        if args.model in ['ae', 'all']:
            ae_results = test_ae(args, device, psnr_level)
            model_results['AE'] = ae_results
            all_results[psnr_level]['AE'] = ae_results

        # 如果测试多个模型，创建比较
        if len(model_results) > 1:
            best_models = compare_models(args, device, psnr_level, model_results)
            all_results[psnr_level]['best_models'] = best_models

        # 创建样本可视化
        if args.model == 'all':
            test_sample_visualization(args, device, psnr_level)

    # 如果测试了所有PSNR级别，创建跨PSNR汇总
    if len(args.psnr_levels) > 1 and args.model == 'all':
        create_psnr_summary(args, all_results)

    # 测试结束时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'=' * 50}")
    print(f"测试完成！总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    print(f"结果已保存至: {args.output_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()