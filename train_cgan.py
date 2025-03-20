import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from models import TargetRadialLengthModule, TargetIdentityModule
from cgan_models import Generator, Discriminator
from hrrp_dataset import HRRPDataset
import random


def add_noise(hrrp_data, noise_level=0.1):
    """
    向HRRP数据添加高斯噪声

    参数:
        hrrp_data (torch.Tensor): 干净的HRRP数据
        noise_level (float): 高斯噪声的标准差

    返回:
        torch.Tensor: 噪声HRRP数据
    """
    noise = torch.randn_like(hrrp_data) * noise_level
    noisy_data = hrrp_data + noise
    # 确保数据保持在有效范围 [0, 1] 内
    noisy_data = torch.clamp(noisy_data, 0, 1)
    return noisy_data


def train_cgan(args):
    """
    训练CGAN用于HRRP数据去噪

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

    # 加载预训练的特征提取器
    G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)
    G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                               num_classes=args.num_classes).to(device)

    # 如果提供，加载权重
    if args.load_feature_extractors:
        G_D.load_state_dict(torch.load(os.path.join(args.load_dir, 'G_D.pth')))
        G_I.load_state_dict(torch.load(os.path.join(args.load_dir, 'G_I.pth')))

    # 将特征提取器设置为评估模式
    G_D.eval()
    G_I.eval()

    # 创建CGAN模型
    generator = Generator(input_dim=args.input_dim,
                          condition_dim=args.feature_dim * 2,
                          hidden_dim=args.hidden_dim).to(device)

    discriminator = Discriminator(input_dim=args.input_dim,
                                  condition_dim=args.feature_dim * 2,
                                  hidden_dim=args.hidden_dim).to(device)

    # 定义损失函数
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.MSELoss()

    # 定义优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # 加载数据集
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 打印关于数据集的一些信息
    sample_data, sample_radial, sample_identity = next(iter(train_loader))
    print(f"样本数据形状: {sample_data.shape}")

    # 准备用于跟踪损失的数组
    d_losses = []
    g_losses = []
    r_losses = []

    # 训练循环
    for epoch in range(args.epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_r_loss = 0

        for i, (clean_data, radial_length, identity_labels) in enumerate(train_loader):
            batch_size = clean_data.shape[0]

            # 将数据移至设备
            clean_data = clean_data.float().to(device)

            # 创建噪声数据
            noisy_data = add_noise(clean_data, noise_level=args.noise_level)

            # 使用预训练模块提取特征
            with torch.no_grad():
                # 提取目标径向长度特征
                f_D, _ = G_D(clean_data)

                # 提取目标身份特征
                f_I, _ = G_I(clean_data)

                # 连接特征以创建条件
                condition = torch.cat([f_D, f_I], dim=1)

            # -----------------------
            # 训练判别器
            # -----------------------
            optimizer_D.zero_grad()

            # 生成伪(去噪)样本
            generated_samples = generator(noisy_data, condition)

            # 准备标签
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # 真实样本的判别器损失
            real_loss = adversarial_loss(discriminator(clean_data, condition), real_labels)

            # 伪样本的判别器损失
            fake_loss = adversarial_loss(discriminator(generated_samples.detach(), condition), fake_labels)

            # 总判别器损失
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            # 训练生成器
            # ---------------------
            optimizer_G.zero_grad()

            # 生成器的对抗损失(欺骗判别器)
            g_adv_loss = adversarial_loss(discriminator(generated_samples, condition), real_labels)

            # 重建损失(生成和干净之间的L1或L2距离)
            g_rec_loss = reconstruction_loss(generated_samples, clean_data)

            # 总生成器损失
            g_loss = g_adv_loss + args.lambda_rec * g_rec_loss
            g_loss.backward()
            optimizer_G.step()

            # 更新epoch损失
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_adv_loss.item()
            epoch_r_loss += g_rec_loss.item()

            # 打印进度
            if i % 10 == 0:
                print(f"[Epoch {epoch + 1}/{args.epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G adv: {g_adv_loss.item():.4f}] "
                      f"[G rec: {g_rec_loss.item():.4f}]")

        # 计算该epoch的平均损失
        epoch_d_loss /= len(train_loader)
        epoch_g_loss /= len(train_loader)
        epoch_r_loss /= len(train_loader)

        # 添加到损失数组
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)
        r_losses.append(epoch_r_loss)

        # 每隔几个epoch保存一次检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 保存生成器和判别器
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, 'generator.pth'))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'discriminator.pth'))

            # 生成并保存样本去噪图像
            if args.save_samples:
                with torch.no_grad():
                    # 从数据集中获取样本
                    sample_idx = np.random.randint(0, len(train_dataset))
                    sample_clean, _, _ = train_dataset[sample_idx]
                    sample_clean = sample_clean.unsqueeze(0).float().to(device)

                    # 创建带噪声的样本
                    sample_noisy = add_noise(sample_clean, noise_level=args.noise_level)

                    # 提取特征
                    f_D, _ = G_D(sample_clean)
                    f_I, _ = G_I(sample_clean)
                    sample_condition = torch.cat([f_D, f_I], dim=1)

                    # 生成去噪样本
                    sample_denoised = generator(sample_noisy, sample_condition)

                    # 绘制并保存结果
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
                    plt.savefig(os.path.join(checkpoint_dir, f'sample_denoising.png'))
                    plt.close()

        # 打印epoch摘要
        print(
            f"Epoch {epoch + 1}/{args.epochs} - D Loss: {epoch_d_loss:.4f}, G Adv Loss: {epoch_g_loss:.4f}, G Rec Loss: {epoch_r_loss:.4f}")

    # 保存最终模型
    torch.save(generator.state_dict(), os.path.join(args.save_dir, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(args.save_dir, 'discriminator_final.pth'))

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(d_losses, label='判别器损失')
    plt.plot(g_losses, label='生成器对抗损失')
    plt.plot(r_losses, label='生成器重建损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'training_loss.png'))
    plt.close()

    print(f"训练完成。模型保存到 {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练用于HRRP去噪的CGAN')
    parser.add_argument('--train_dir', type=str, default='datasets/simulated_3/train',
                        help='包含训练数据的目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints/cgan',
                        help='保存训练模型的目录')
    parser.add_argument('--load_dir', type=str, default='checkpoints',
                        help='加载预训练特征提取器的目录')
    parser.add_argument('--load_feature_extractors', action='store_true',
                        help='是否加载预训练特征提取器')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='训练的批量大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练的epoch数')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='学习率')
    parser.add_argument('--input_dim', type=int, default=500,
                        help='输入HRRP序列的维度')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='特征提取器输出的维度')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='CGAN中隐藏层的维度')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='目标身份类别的数量')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='添加到干净样本的高斯噪声的标准差')
    parser.add_argument('--lambda_rec', type=float, default=10.0,
                        help='生成器损失中重建损失的权重')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='保存检查点的epoch间隔')
    parser.add_argument('--save_samples', default=1,
                        help='是否保存样本去噪结果')
    parser.add_argument('--seed', type=int, default=42,
                        help='可重现性的随机种子')

    args = parser.parse_args()

    # 如果不存在，创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练CGAN
    train_cgan(args)