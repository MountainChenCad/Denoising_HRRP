# train_ae.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from ae_models import AutoEncoder
from hrrp_dataset import HRRPDataset
import random


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


def train_ae(args):
    """
    训练用于HRRP信号去噪的AE

    参数:
        args: 训练参数
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置随机种子以确保可重现性
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 创建AE模型
    model = AutoEncoder(input_dim=args.input_dim,
                        latent_dim=args.latent_dim,
                        hidden_dim=args.hidden_dim).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 加载数据集
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 打印关于数据集和模型的信息
    sample_data, _, _ = next(iter(train_loader))
    print(f"样本数据形状: {sample_data.shape}")
    print(f"模型概览: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params}")

    # 用于跟踪损失的数组
    train_losses = []

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for i, (clean_data, _, _) in enumerate(train_loader):
            # 将数据移至设备
            clean_data = clean_data.float().to(device)

            # 创建噪声数据
            noisy_data = add_noise(clean_data, noise_level=args.noise_level)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播(去噪任务: noisy->clean)
            reconstructed, _ = model(noisy_data)

            # 计算损失(重建和干净数据之间的损失)
            loss = criterion(reconstructed, clean_data)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 更新epoch损失
            epoch_loss += loss.item()

            # 打印批次进度
            if i % 10 == 0:
                print(f"[Epoch {epoch + 1}/{args.epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[Loss: {loss.item():.4f}]")

        # 计算平均epoch损失
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # 打印epoch摘要
        print(f"Epoch {epoch + 1}/{args.epochs} - Avg Loss: {epoch_loss:.4f}")

        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 保存模型
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'ae_model.pth'))

            # 保存样本去噪结果
            if args.save_samples:
                model.eval()
                with torch.no_grad():
                    # 从数据集中获取样本
                    sample_idx = np.random.randint(0, len(train_dataset))
                    sample_clean, _, _ = train_dataset[sample_idx]
                    sample_clean = sample_clean.unsqueeze(0).float().to(device)

                    # 创建带噪声的样本
                    sample_noisy = add_noise(sample_clean, noise_level=args.noise_level)

                    # 对样本进行去噪
                    sample_denoised, _ = model(sample_noisy)

                    # 绘制结果
                    plt.figure(figsize=(12, 4))

                    plt.subplot(1, 3, 1)
                    plt.plot(sample_clean.cpu().numpy()[0])
                    plt.title('Clean HRRP')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.subplot(1, 3, 2)
                    plt.plot(sample_noisy.cpu().numpy()[0])
                    plt.title(f'Noisy HRRP (sigma={args.noise_level})')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.subplot(1, 3, 3)
                    plt.plot(sample_denoised.cpu().numpy()[0])
                    plt.title('Denoised HRRP')
                    plt.xlabel('Range')
                    plt.ylabel('Magnitude')

                    plt.tight_layout()
                    plt.savefig(os.path.join(checkpoint_dir, 'sample_denoising.png'))
                    plt.close()

                model.train()

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'ae_model_final.pth'))

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('AE Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'training_loss.png'))
    plt.close()

    print(f"训练完成。模型保存到 {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练用于HRRP去噪的AE')
    parser.add_argument('--train_dir', type=str, default='datasets/simulated_3/train',
                        help='包含训练数据的目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints/ae',
                        help='保存训练模型的目录')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='训练的批量大小')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练的epoch数')
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='学习率')
    parser.add_argument('--input_dim', type=int, default=500,
                        help='输入HRRP序列的维度')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='潜在空间的维度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='隐藏层的维度')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='添加到干净样本的高斯噪声的标准差')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='保存检查点的epoch间隔')
    parser.add_argument('--save_samples', action='store_true',
                        help='是否保存样本去噪结果')
    parser.add_argument('--seed', type=int, default=42,
                        help='可重现性的随机种子')

    args = parser.parse_args()

    # 如果不存在，创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练AE
    train_ae(args)