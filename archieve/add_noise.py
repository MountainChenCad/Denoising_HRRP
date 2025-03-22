# compare_methods.py (修改版，增加了AE和多PSNR比较)
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from models.modules import TargetRadialLengthModule, TargetIdentityModule
from models.cgan_models import Generator
from models.cae_models import ConvAutoEncoder
from models.ae_models import AutoEncoder
from utils.hrrp_dataset import HRRPDataset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim


def add_noise_for_psnr(signal, psnr_db):
    """
    添加高斯噪声以达到特定的PSNR(峰值信噪比)值

    参数:
        signal (torch.Tensor): 原始干净信号
        psnr_db (float): 期望的PSNR值(dB)

    返回:
        torch.Tensor: 添加噪声后的信号
    """
    # 计算信号功率
    signal_power = torch.mean(signal ** 2)

    # 根据期望的PSNR计算噪声功率
    noise_power = signal_power / (10 ** (psnr_db / 10))

    # 生成高斯噪声
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)

    # 将噪声添加到信号中
    noisy_signal = signal + noise

    # 确保数据保持在有效范围 [0, 1] 内
    noisy_signal = torch.clamp(noisy_signal, 0, 1)

    return noisy_signal


def calculate_ssim(x, y):
    """
    计算两个1D信号之间的SSIM

    参数:
        x (numpy.ndarray): 第一个信号 (1D)
        y (numpy.ndarray): 第二个信号 (1D)

    返回:
        float: SSIM值在0到1之间 (越高越好)
    """
    # SSIM要求输入为非负
    return ssim(x, y, data_range=1.0)


def calculate_psnr(original, noisy):
    """
    计算PSNR

    参数:
        original (torch.Tensor): 原始干净信号
        noisy (torch.Tensor): 噪声或恢复信号

    返回:
        float: PSNR值(dB)
    """
    mse = torch.mean((original - noisy) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def compare_methods(args, psnr_level):
    """
    比较CGAN、CAE和AE方法用于HRRP信号去噪，在特定PSNR级别下

    参数:
        args: 比较参数
        psnr_level: 当前测试的PSNR级别(dB)
    """
    # 创建特定PSNR级别的输出目录
    output_dir = os.path.join(args.output_dir, f"psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} 进行 PSNR = {psnr_level}dB 的比较")

    # 加载CGAN模型
    # 特征提取器
    G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)
    G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                               num_classes=args.num_classes).to(device)

    # 加载其权重
    G_D.load_state_dict(torch.load(os.path.join(args.cgan_dir, 'G_D_final.pth')))
    G_I.load_state_dict(torch.load(os.path.join(args.cgan_dir, 'G_I_final.pth')))

    # 将特征提取器设置为评估模式
    G_D.eval()
    G_I.eval()

    # 加载CGAN生成器
    cgan_generator = Generator(input_dim=args.input_dim,
                               condition_dim=args.feature_dim * 2,
                               hidden_dim=args.hidden_dim).to(device)

    # 加载生成器权重
    cgan_generator.load_state_dict(torch.load(os.path.join(args.cgan_dir, 'generator_final.pth')))
    cgan_generator.eval()

    # 加载CAE模型
    cae_model = ConvAutoEncoder(input_dim=args.input_dim,
                                latent_dim=args.latent_dim,
                                hidden_dim=args.hidden_dim).to(device)

    # 加载CAE权重
    cae_model.load_state_dict(torch.load(os.path.join(args.cae_dir, 'cae_model_final.pth')))
    cae_model.eval()

    # 加载AE模型
    ae_model = AutoEncoder(input_dim=args.input_dim,
                           latent_dim=args.latent_dim,
                           hidden_dim=args.ae_hidden_dim).to(device)

    # 加载AE权重
    ae_model.load_state_dict(torch.load(os.path.join(args.ae_dir, 'ae_model_final.pth')))
    ae_model.eval()

    # 加载测试数据集
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义用于评估的损失函数
    mse_loss = nn.MSELoss()

    # 比较去噪性能
    cgan_total_mse = 0
    cae_total_mse = 0
    ae_total_mse = 0
    cgan_total_ssim = 0
    cae_total_ssim = 0
    ae_total_ssim = 0
    cgan_total_psnr = 0
    cae_total_psnr = 0
    ae_total_psnr = 0
    noisy_total_psnr = 0

    with torch.no_grad():
        for i, (clean_data, radial_length, identity_label) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # 将数据移至设备
            clean_data = clean_data.float().to(device)

            # 创建噪声数据，基于指定的PSNR级别
            noisy_data = add_noise_for_psnr(clean_data, psnr_level)

            # 提取CGAN条件特征
            f_D, _ = G_D(clean_data)
            f_I, _ = G_I(clean_data)
            condition = torch.cat([f_D, f_I], dim=1)

            # 使用CGAN生成去噪数据
            cgan_denoised = cgan_generator(noisy_data, condition)

            # 使用CAE生成去噪数据
            cae_denoised, _ = cae_model(noisy_data)

            # 使用AE生成去噪数据
            ae_denoised, _ = ae_model(noisy_data)

            # 计算所有方法的MSE
            cgan_mse = mse_loss(cgan_denoised, clean_data).item()
            cae_mse = mse_loss(cae_denoised, clean_data).item()
            ae_mse = mse_loss(ae_denoised, clean_data).item()

            # 计算所有方法的PSNR
            noisy_psnr = calculate_psnr(clean_data, noisy_data)
            cgan_psnr = calculate_psnr(clean_data, cgan_denoised)
            cae_psnr = calculate_psnr(clean_data, cae_denoised)
            ae_psnr = calculate_psnr(clean_data, ae_denoised)

            # 计算所有方法的SSIM
            clean_np = clean_data.cpu().numpy()[0]
            noisy_np = noisy_data.cpu().numpy()[0]
            cgan_np = cgan_denoised.cpu().numpy()[0]
            cae_np = cae_denoised.cpu().numpy()[0]
            ae_np = ae_denoised.cpu().numpy()[0]

            cgan_ssim = calculate_ssim(clean_np, cgan_np)
            cae_ssim = calculate_ssim(clean_np, cae_np)
            ae_ssim = calculate_ssim(clean_np, ae_np)

            # 累积指标
            cgan_total_mse += cgan_mse
            cae_total_mse += cae_mse
            ae_total_mse += ae_mse
            cgan_total_ssim += cgan_ssim
            cae_total_ssim += cae_ssim
            ae_total_ssim += ae_ssim
            noisy_total_psnr += noisy_psnr
            cgan_total_psnr += cgan_psnr
            cae_total_psnr += cae_psnr
            ae_total_psnr += ae_psnr

            # 绘制比较结果
            plt.figure(figsize=(18, 4))

            plt.subplot(1, 5, 1)
            plt.plot(clean_data.cpu().numpy()[0])
            plt.title('Clean HRRP')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')

            plt.subplot(1, 5, 2)
            plt.plot(noisy_data.cpu().numpy()[0])
            plt.title(f'Noisy HRRP\nPSNR: {noisy_psnr:.2f}dB')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')

            plt.subplot(1, 5, 3)
            plt.plot(cgan_denoised.cpu().numpy()[0])
            plt.title(f'CGAN Denoised\nPSNR: {cgan_psnr:.2f}dB, SSIM: {cgan_ssim:.4f}')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')

            plt.subplot(1, 5, 4)
            plt.plot(cae_denoised.cpu().numpy()[0])
            plt.title(f'CAE Denoised\nPSNR: {cae_psnr:.2f}dB, SSIM: {cae_ssim:.4f}')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')

            plt.subplot(1, 5, 5)
            plt.plot(ae_denoised.cpu().numpy()[0])
            plt.title(f'AE Denoised\nPSNR: {ae_psnr:.2f}dB, SSIM: {ae_ssim:.4f}')
            plt.xlabel('Range Bin')
            plt.ylabel('Magnitude')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'comparison_sample_{i + 1}.png'))
            plt.close()

            print(f"PSNR={psnr_level}dB, 样本 {i + 1}:")
            print(f"  噪声信号 - PSNR: {noisy_psnr:.2f}dB")
            print(f"  CGAN - PSNR: {cgan_psnr:.2f}dB, SSIM: {cgan_ssim:.4f}")
            print(f"  CAE  - PSNR: {cae_psnr:.2f}dB, SSIM: {cae_ssim:.4f}")
            print(f"  AE   - PSNR: {ae_psnr:.2f}dB, SSIM: {ae_ssim:.4f}")

            # 基于两个指标找出最佳方法
            methods = ["CGAN", "CAE", "AE"]
            psnr_values = [cgan_psnr, cae_psnr, ae_psnr]
            ssim_values = [cgan_ssim, cae_ssim, ae_ssim]

            best_psnr_idx = np.argmax(psnr_values)
            best_ssim_idx = np.argmax(ssim_values)

            print(f"  最佳PSNR: {methods[best_psnr_idx]} ({psnr_values[best_psnr_idx]:.2f}dB)")
            print(f"  最佳SSIM: {methods[best_ssim_idx]} ({ssim_values[best_ssim_idx]:.4f})")
            print()

    # 计算平均指标
    n_samples = min(args.num_samples, len(test_loader))
    avg_noisy_psnr = noisy_total_psnr / n_samples
    avg_cgan_psnr = cgan_total_psnr / n_samples
    avg_cae_psnr = cae_total_psnr / n_samples
    avg_ae_psnr = ae_total_psnr / n_samples

    avg_cgan_mse = cgan_total_mse / n_samples
    avg_cae_mse = cae_total_mse / n_samples
    avg_ae_mse = ae_total_mse / n_samples

    avg_cgan_ssim = cgan_total_ssim / n_samples
    avg_cae_ssim = cae_total_ssim / n_samples
    avg_ae_ssim = ae_total_ssim / n_samples

    print(f"\nPSNR={psnr_level}dB, 平均指标:")
    print(f"  噪声信号 - PSNR: {avg_noisy_psnr:.2f}dB")
    print(f"  CGAN - PSNR: {avg_cgan_psnr:.2f}dB, MSE: {avg_cgan_mse:.4f}, SSIM: {avg_cgan_ssim:.4f}")
    print(f"  CAE  - PSNR: {avg_cae_psnr:.2f}dB, MSE: {avg_cae_mse:.4f}, SSIM: {avg_cae_ssim:.4f}")
    print(f"  AE   - PSNR: {avg_ae_psnr:.2f}dB, MSE: {avg_ae_mse:.4f}, SSIM: {avg_ae_ssim:.4f}")

    # 计算PSNR改善量
    cgan_psnr_improvement = avg_cgan_psnr - avg_noisy_psnr
    cae_psnr_improvement = avg_cae_psnr - avg_noisy_psnr
    ae_psnr_improvement = avg_ae_psnr - avg_noisy_psnr

    print(f"\nPSNR改善量:")
    print(f"  CGAN: +{cgan_psnr_improvement:.2f}dB")
    print(f"  CAE: +{cae_psnr_improvement:.2f}dB")
    print(f"  AE: +{ae_psnr_improvement:.2f}dB")

    # 找出总体最佳方法
    methods = ["CGAN", "CAE", "AE"]
    avg_psnr_values = [avg_cgan_psnr, avg_cae_psnr, avg_ae_psnr]
    avg_ssim_values = [avg_cgan_ssim, avg_cae_ssim, avg_ae_ssim]

    best_psnr_idx = np.argmax(avg_psnr_values)
    best_ssim_idx = np.argmax(avg_ssim_values)

    print(f"\n按PSNR最佳方法: {methods[best_psnr_idx]} ({avg_psnr_values[best_psnr_idx]:.2f}dB)")
    print(f"按SSIM最佳方法: {methods[best_ssim_idx]} ({avg_ssim_values[best_ssim_idx]:.4f})")

    # 保存摘要结果
    with open(os.path.join(output_dir, 'comparison_results.txt'), 'w') as f:
        f.write(f"CGAN vs CAE vs AE 在PSNR={psnr_level}dB条件下的比较结果\n")
        f.write(f"=======================================================\n\n")
        f.write(f"测试样本数: {n_samples}\n\n")
        f.write(f"平均指标:\n")
        f.write(f"  噪声信号 - PSNR: {avg_noisy_psnr:.2f}dB\n")
        f.write(f"  CGAN - PSNR: {avg_cgan_psnr:.2f}dB, MSE: {avg_cgan_mse:.4f}, SSIM: {avg_cgan_ssim:.4f}\n")
        f.write(f"  CAE  - PSNR: {avg_cae_psnr:.2f}dB, MSE: {avg_cae_mse:.4f}, SSIM: {avg_cae_ssim:.4f}\n")
        f.write(f"  AE   - PSNR: {avg_ae_psnr:.2f}dB, MSE: {avg_ae_mse:.4f}, SSIM: {avg_ae_ssim:.4f}\n\n")
        f.write(f"PSNR改善量:\n")
        f.write(f"  CGAN: +{cgan_psnr_improvement:.2f}dB\n")
        f.write(f"  CAE: +{cae_psnr_improvement:.2f}dB\n")
        f.write(f"  AE: +{ae_psnr_improvement:.2f}dB\n\n")
        f.write(f"按PSNR最佳方法: {methods[best_psnr_idx]} ({avg_psnr_values[best_psnr_idx]:.2f}dB)\n")
        f.write(f"按SSIM最佳方法: {methods[best_ssim_idx]} ({avg_ssim_values[best_ssim_idx]:.4f})\n")

    # 创建比较方法的条形图
    plt.figure(figsize=(15, 12))

    # PSNR比较(越高越好)
    plt.subplot(3, 1, 1)
    methods = ['Noisy', 'CGAN', 'CAE', 'AE']
    psnr_values = [avg_noisy_psnr, avg_cgan_psnr, avg_cae_psnr, avg_ae_psnr]
    improvement_values = [0, cgan_psnr_improvement, cae_psnr_improvement, ae_psnr_improvement]

    bars = plt.bar(methods, psnr_values, color=['gray', 'blue', 'green', 'red'])
    plt.title(f'PSNR Comparison at {psnr_level}dB Noise Level', fontsize=14)
    plt.xlabel('Methods', fontsize=12)
    plt.ylabel('Average PSNR (dB)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在条形上方添加PSNR值和改善量
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == 0:  # 噪声信号，没有改善
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.2f}dB', ha='center', va='bottom', fontsize=10)
        else:  # 去噪方法，显示改善量
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.2f}dB\n(+{improvement_values[i]:.2f}dB)',
                     ha='center', va='bottom', fontsize=10)

    # MSE比较(越低越好)
    plt.subplot(3, 1, 2)
    methods = ['CGAN', 'CAE', 'AE']
    mse_values = [avg_cgan_mse, avg_cae_mse, avg_ae_mse]
    bars = plt.bar(methods, mse_values, color=['blue', 'green', 'red'])
    plt.title('MSE Comparison', fontsize=14)
    plt.xlabel('Methods', fontsize=12)
    plt.ylabel('Average MSE', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在条形上方添加文本标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.0005,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # SSIM比较(越高越好)
    plt.subplot(3, 1, 3)
    methods = ['CGAN', 'CAE', 'AE']
    ssim_values = [avg_cgan_ssim, avg_cae_ssim, avg_ae_ssim]
    bars = plt.bar(methods, ssim_values, color=['blue', 'green', 'red'])
    plt.title('SSIM Comparison', fontsize=14)
    plt.xlabel('Methods', fontsize=12)
    plt.ylabel('Average SSIM', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在条形上方添加文本标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'))
    plt.close()

    return avg_cgan_psnr, avg_cae_psnr, avg_ae_psnr, avg_cgan_ssim, avg_cae_ssim, avg_ae_ssim, cgan_psnr_improvement, cae_psnr_improvement, ae_psnr_improvement


def main():
    parser = argparse.ArgumentParser(description='比较CGAN, CAE和AE用于HRRP去噪')
    parser.add_argument('--test_dir', type=str, default='datasets/simulated_3/test',
                        help='包含测试数据的目录')
    parser.add_argument('--cgan_dir', type=str, default='checkpoints/cgan',
                        help='包含训练好的CGAN模型的目录')
    parser.add_argument('--cae_dir', type=str, default='checkpoints/cae',
                        help='包含训练好的CAE模型的目录')
    parser.add_argument('--ae_dir', type=str, default='checkpoints/ae',
                        help='包含训练好的AE模型的目录')
    parser.add_argument('--output_dir', type=str, default='results/comparison',
                        help='保存比较结果的目录')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='要处理的测试样本数')
    parser.add_argument('--input_dim', type=int, default=500,
                        help='输入HRRP序列的维度')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='CGAN的特征维度')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='CAE和AE的潜在维度')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='CGAN和CAE的隐藏维度')
    parser.add_argument('--ae_hidden_dim', type=int, default=256,
                        help='AE的隐藏维度')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='目标身份类别的数量')

    args = parser.parse_args()

    # 如果不存在，创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 定义要测试的PSNR级别
    psnr_levels = [20, 10, 0]

    # 存储每个PSNR级别的结果
    results = {}

    # 依次在每个PSNR级别上执行比较
    for psnr_level in psnr_levels:
        print(f"\n{'=' * 50}")
        print(f"开始在PSNR = {psnr_level}dB条件下比较去噪方法")
        print(f"{'=' * 50}\n")

        results[psnr_level] = compare_methods(args, psnr_level)

    # 创建所有PSNR级别的汇总比较图
    plt.figure(figsize=(18, 10))

    # PSNR提升比较
    plt.subplot(1, 2, 1)
    x = range(len(psnr_levels))
    width = 0.25

    # 获取各PSNR级别下的PSNR提升值
    cgan_improvements = [results[level][6] for level in psnr_levels]  # index 6 for cgan_psnr_improvement
    cae_improvements = [results[level][7] for level in psnr_levels]  # index 7 for cae_psnr_improvement
    ae_improvements = [results[level][8] for level in psnr_levels]  # index 8 for ae_psnr_improvement

    plt.bar([p - width for p in x], cgan_improvements, width, label='CGAN', color='blue')
    plt.bar(x, cae_improvements, width, label='CAE', color='green')
    plt.bar([p + width for p in x], ae_improvements, width, label='AE', color='red')

    plt.xlabel('Input Noise Level (PSNR in dB)', fontsize=12)
    plt.ylabel('PSNR Improvement (dB)', fontsize=12)
    plt.title('PSNR Improvement vs. Noise Level', fontsize=14)
    plt.xticks(x, [f'{level}dB' for level in psnr_levels])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # SSIM比较
    plt.subplot(1, 2, 2)

    # 获取各PSNR级别下的SSIM值
    cgan_ssim = [results[level][3] for level in psnr_levels]  # index 3 for avg_cgan_ssim
    cae_ssim = [results[level][4] for level in psnr_levels]  # index 4 for avg_cae_ssim
    ae_ssim = [results[level][5] for level in psnr_levels]  # index 5 for avg_ae_ssim

    plt.bar([p - width for p in x], cgan_ssim, width, label='CGAN', color='blue')
    plt.bar(x, cae_ssim, width, label='CAE', color='green')
    plt.bar([p + width for p in x], ae_ssim, width, label='AE', color='red')

    plt.xlabel('Input Noise Level (PSNR in dB)', fontsize=12)
    plt.ylabel('Average SSIM', fontsize=12)
    plt.title('SSIM vs. Noise Level', fontsize=14)
    plt.xticks(x, [f'{level}dB' for level in psnr_levels])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'psnr_comparison_summary.png'))
    plt.close()

    # 保存汇总结果到文本文件
    with open(os.path.join(args.output_dir, 'summary_results.txt'), 'w') as f:
        f.write(f"CGAN vs CAE vs AE 在不同PSNR条件下的比较结果汇总\n")
        f.write(f"=================================================\n\n")

        for level in psnr_levels:
            avg_cgan_psnr, avg_cae_psnr, avg_ae_psnr, avg_cgan_ssim, avg_cae_ssim, avg_ae_ssim, cgan_imp, cae_imp, ae_imp = \
            results[level]

            f.write(f"PSNR = {level}dB 条件下的结果:\n")
            f.write(f"  CGAN - PSNR: {avg_cgan_psnr:.2f}dB, SSIM: {avg_cgan_ssim:.4f}, 改善: +{cgan_imp:.2f}dB\n")
            f.write(f"  CAE  - PSNR: {avg_cae_psnr:.2f}dB, SSIM: {avg_cae_ssim:.4f}, 改善: +{cae_imp:.2f}dB\n")
            f.write(f"  AE   - PSNR: {avg_ae_psnr:.2f}dB, SSIM: {avg_ae_ssim:.4f}, 改善: +{ae_imp:.2f}dB\n\n")

            # 找出最佳方法
            methods = ["CGAN", "CAE", "AE"]
            psnr_values = [avg_cgan_psnr, avg_cae_psnr, avg_ae_psnr]
            improvement_values = [cgan_imp, cae_imp, ae_imp]
            ssim_values = [avg_cgan_ssim, avg_cae_ssim, avg_ae_ssim]

            best_psnr_idx = np.argmax(psnr_values)
            best_imp_idx = np.argmax(improvement_values)
            best_ssim_idx = np.argmax(ssim_values)

            f.write(f"  按PSNR值最佳方法: {methods[best_psnr_idx]} ({psnr_values[best_psnr_idx]:.2f}dB)\n")
            f.write(f"  按PSNR改善最佳方法: {methods[best_imp_idx]} (+{improvement_values[best_imp_idx]:.2f}dB)\n")
            f.write(f"  按SSIM最佳方法: {methods[best_ssim_idx]} ({ssim_values[best_ssim_idx]:.4f})\n")
            f.write(f"{'=' * 50}\n\n")

        # 输出总体结论
        f.write("总体结论:\n")
        # 计算每种方法在多少个PSNR级别上是最好的
        psnr_winners = {}
        ssim_winners = {}

        for level in psnr_levels:
            _, _, _, avg_cgan_ssim, avg_cae_ssim, avg_ae_ssim, cgan_imp, cae_imp, ae_imp = results[level]

            # 按PSNR改善找最佳
            imp_values = [cgan_imp, cae_imp, ae_imp]
            methods = ["CGAN", "CAE", "AE"]
            best_imp_idx = np.argmax(imp_values)
            best_method = methods[best_imp_idx]

            psnr_winners[best_method] = psnr_winners.get(best_method, 0) + 1

            # 按SSIM找最佳
            ssim_values = [avg_cgan_ssim, avg_cae_ssim, avg_ae_ssim]
            best_ssim_idx = np.argmax(ssim_values)
            best_method = methods[best_ssim_idx]

            ssim_winners[best_method] = ssim_winners.get(best_method, 0) + 1

        f.write(f"  按PSNR改善计算，各方法获胜次数:\n")
        for method, count in psnr_winners.items():
            f.write(f"    {method}: {count}/{len(psnr_levels)} 次\n")

        f.write(f"\n  按SSIM计算，各方法获胜次数:\n")
        for method, count in ssim_winners.items():
            f.write(f"    {method}: {count}/{len(psnr_levels)} 次\n")

    print(f"\n所有比较完成，汇总结果已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()