# test_ae.py
import torch
import torch.nn as nn
import os
import argparse
import matplotlib.pyplot as plt
from models.ae_models import AutoEncoder
from utils.hrrp_dataset import HRRPDataset
from torch.utils.data import DataLoader


def add_noise(hrrp_data, noise_level=0.1):
    """
    向HRRP数据添加高斯噪声

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


def test_ae(args):
    """
    测试用于HRRP信号去噪的AE

    参数:
        args: 测试参数
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    model = AutoEncoder(input_dim=args.input_dim,
                        latent_dim=args.latent_dim,
                        hidden_dim=args.hidden_dim).to(device)

    # 加载训练权重
    model.load_state_dict(torch.load(os.path.join(args.load_dir, 'ae_model_final.pth')))
    model.eval()

    # 加载测试数据集
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 定义用于评估的损失函数
    mse_loss = nn.MSELoss()

    # 对测试样本进行去噪
    total_mse_noisy = 0
    total_mse_denoised = 0

    with torch.no_grad():
        for i, (clean_data, radial_length, identity_label) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # 将数据移至设备
            clean_data = clean_data.float().to(device)

            # 创建噪声数据
            noisy_data = add_noise(clean_data, noise_level=args.noise_level)

            # 生成去噪数据
            denoised_data, _ = model(noisy_data)

            # 计算噪声和去噪数据的MSE
            mse_noisy = mse_loss(noisy_data, clean_data).item()
            mse_denoised = mse_loss(denoised_data, clean_data).item()

            total_mse_noisy += mse_noisy
            total_mse_denoised += mse_denoised

            # 绘制结果
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

            print(f"样本 {i + 1}:")
            print(f"  噪声 MSE: {mse_noisy:.4f}")
            print(f"  去噪 MSE: {mse_denoised:.4f}")
            print(f"  改进: {(mse_noisy - mse_denoised) / mse_noisy * 100:.2f}%")

    # 计算平均MSE
    avg_mse_noisy = total_mse_noisy / min(args.num_samples, len(test_loader))
    avg_mse_denoised = total_mse_denoised / min(args.num_samples, len(test_loader))

    print(f"\n平均噪声 MSE: {avg_mse_noisy:.4f}")
    print(f"平均去噪 MSE: {avg_mse_denoised:.4f}")
    print(f"平均改进: {(avg_mse_noisy - avg_mse_denoised) / avg_mse_noisy * 100:.2f}%")

    # 保存摘要结果
    with open(os.path.join(args.output_dir, 'results_summary.txt'), 'w') as f:
        f.write(f"AE去噪结果\n")
        f.write(f"===================\n\n")
        f.write(f"噪声级别: {args.noise_level}\n")
        f.write(f"平均噪声MSE: {avg_mse_noisy:.4f}\n")
        f.write(f"平均去噪MSE: {avg_mse_denoised:.4f}\n")
        f.write(f"平均改进: {(avg_mse_noisy - avg_mse_denoised) / avg_mse_noisy * 100:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试用于HRRP去噪的AE')
    parser.add_argument('--test_dir', type=str, default='datasets/simulated_3/test',
                        help='包含测试数据的目录')
    parser.add_argument('--load_dir', type=str, default='checkpoints/ae',
                        help='加载训练模型的目录')
    parser.add_argument('--output_dir', type=str, default='results/ae',
                        help='保存测试结果的目录')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='要处理的测试样本数')
    parser.add_argument('--input_dim', type=int, default=500,
                        help='输入HRRP序列的维度')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='潜在空间的维度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='隐藏层的维度')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='添加到干净样本的高斯噪声的标准差')

    args = parser.parse_args()

    # 测试AE
    test_ae(args)