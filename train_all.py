# train_all.py - 统一训练入口
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import random
import time

# 导入模型定义
from models import TargetRadialLengthModule, TargetIdentityModule
from cgan_models import Generator, Discriminator
from cae_models import ConvAutoEncoder
from ae_models import AutoEncoder
from hrrp_dataset import HRRPDataset
from noise_utils import add_noise_for_psnr


def setup_seed(seed):
    """设置随机种子以确保可重现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_modules(args, device, psnr_level=None):
    """
    训练目标辐射长度模块(G_D)和目标身份模块(G_I)

    参数:
        args: 命令行参数
        device: 计算设备(CPU/GPU)
        psnr_level: 不适用于模块训练，可忽略
    返回:
        字典，包含训练历史和模型路径
    """
    print(f"\n{'-' * 20} 开始训练特征提取器模块 {'-' * 20}")

    # 创建保存路径
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练结果和历史记录
    results = {
        "G_D": {"loss_history": []},
        "G_I": {"loss_history": [], "acc_history": []},
        "model_paths": {}
    }

    # 加载数据集
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 根据数据集更新类别数量
    num_classes = train_dataset.get_num_classes()
    print(f"数据集中的类别数量: {num_classes}")

    # 训练G_D模块（如果指定）
    if args.module in ['G_D', 'both']:
        print(f"\n{'-' * 10} 训练目标辐射长度模块(G_D) {'-' * 10}")

        # 创建G_D模型
        G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(G_D.parameters(), lr=args.lr)

        # 训练循环
        G_D.train()
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            start_time = time.time()

            for i, (data, radial_length, _) in enumerate(train_loader):
                # 移动数据到设备
                data = data.float().to(device)
                radial_length = radial_length.float().to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                _, predicted_radial_length = G_D(data)

                # 计算损失
                loss = criterion(predicted_radial_length, radial_length)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 更新累计损失
                epoch_loss += loss.item()

                # 打印进度
                if i % 10 == 0:
                    print(
                        f"[Epoch {epoch + 1}/{args.epochs}] [Batch {i}/{len(train_loader)}] [Loss: {loss.item():.6f}]")

            # 计算平均epoch损失
            avg_epoch_loss = epoch_loss / len(train_loader)
            results["G_D"]["loss_history"].append(avg_epoch_loss)

            # 打印epoch摘要
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{args.epochs} - Avg Loss: {avg_epoch_loss:.6f} - Time: {epoch_time:.2f}s")

            # 定期保存检查点
            if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                checkpoint_path = os.path.join(args.save_dir, f"G_D_epoch_{epoch + 1}.pth")
                torch.save(G_D.state_dict(), checkpoint_path)

        # 保存最终模型
        final_model_path = os.path.join(args.save_dir, "G_D.pth")
        torch.save(G_D.state_dict(), final_model_path)
        results["model_paths"]["G_D"] = final_model_path
        print(f"G_D模型已保存至 {final_model_path}")

        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(results["G_D"]["loss_history"])
        plt.title("G_D Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        loss_plot_path = os.path.join(args.save_dir, "G_D_loss.png")
        plt.savefig(loss_plot_path)
        plt.close()

    # 训练G_I模块（如果指定）
    if args.module in ['G_I', 'both']:
        print(f"\n{'-' * 10} 训练目标身份模块(G_I) {'-' * 10}")

        # 创建G_I模型
        G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                                   num_classes=num_classes).to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(G_I.parameters(), lr=args.lr)

        # 训练循环
        G_I.train()
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()

            for i, (data, _, identity_labels) in enumerate(train_loader):
                # 移动数据到设备
                data = data.float().to(device)
                identity_labels = identity_labels.long().to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                _, identity_logits = G_I(data)

                # 计算损失
                loss = criterion(identity_logits, identity_labels)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 更新累计损失
                epoch_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(identity_logits.data, 1)
                total += identity_labels.size(0)
                correct += (predicted == identity_labels).sum().item()

                # 打印进度
                if i % 10 == 0:
                    print(
                        f"[Epoch {epoch + 1}/{args.epochs}] [Batch {i}/{len(train_loader)}] [Loss: {loss.item():.6f}]")

            # 计算平均epoch损失和准确率
            avg_epoch_loss = epoch_loss / len(train_loader)
            accuracy = 100 * correct / total
            results["G_I"]["loss_history"].append(avg_epoch_loss)
            results["G_I"]["acc_history"].append(accuracy)

            # 打印epoch摘要
            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{args.epochs} - Avg Loss: {avg_epoch_loss:.6f} - Accuracy: {accuracy:.2f}% - Time: {epoch_time:.2f}s")

            # 定期保存检查点
            if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                checkpoint_path = os.path.join(args.save_dir, f"G_I_epoch_{epoch + 1}.pth")
                torch.save(G_I.state_dict(), checkpoint_path)

        # 保存最终模型
        final_model_path = os.path.join(args.save_dir, "G_I.pth")
        torch.save(G_I.state_dict(), final_model_path)
        results["model_paths"]["G_I"] = final_model_path
        print(f"G_I模型已保存至 {final_model_path}")

        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(results["G_I"]["loss_history"])
        plt.title("G_I Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        loss_plot_path = os.path.join(args.save_dir, "G_I_loss.png")
        plt.savefig(loss_plot_path)
        plt.close()

        # 绘制准确率曲线
        plt.figure(figsize=(10, 5))
        plt.plot(results["G_I"]["acc_history"])
        plt.title("G_I Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        acc_plot_path = os.path.join(args.save_dir, "G_I_accuracy.png")
        plt.savefig(acc_plot_path)
        plt.close()

    return results


def train_cgan(args, device, psnr_level):
    """
    训练CGAN用于HRRP去噪

    参数:
        args: 命令行参数
        device: 计算设备(CPU/GPU)
        psnr_level: 目标PSNR级别(dB)
    返回:
        字典，包含训练历史和模型路径
    """
    print(f"\n{'-' * 20} 开始训练CGAN (PSNR={psnr_level}dB) {'-' * 20}")

    # 为当前PSNR级别创建保存目录
    psnr_save_dir = os.path.join(args.save_dir, f"cgan_psnr_{psnr_level}dB")
    os.makedirs(psnr_save_dir, exist_ok=True)

    # 训练结果和历史记录
    results = {
        "loss_history": {
            "discriminator": [],
            "generator_adv": [],
            "generator_rec": [],
            "G_D": [],
            "G_I": []
        },
        "model_paths": {}
    }

    # 加载特征提取器
    G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)
    G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                               num_classes=args.num_classes).to(device)

    # 如果指定，加载预训练权重
    if args.load_feature_extractors:
        G_D.load_state_dict(torch.load(os.path.join(args.load_dir, 'G_D.pth')))
        G_I.load_state_dict(torch.load(os.path.join(args.load_dir, 'G_I.pth')))
        print("已加载预训练特征提取器")

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
    regression_loss = nn.MSELoss()  # 用于G_D的回归损失
    classification_loss = nn.CrossEntropyLoss()  # 用于G_I的分类损失

    # 分离优化器 - 为每个模型使用单独的优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_GD = optim.Adam(G_D.parameters(), lr=args.lr_gd, betas=(0.5, 0.999))
    optimizer_GI = optim.Adam(G_I.parameters(), lr=args.lr_gi, betas=(0.5, 0.999))

    # 加载数据集
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 获取数据集的类别数量
    num_classes = train_dataset.get_num_classes()
    print(f"数据集中的类别数量: {num_classes}")

    # 训练循环
    for epoch in range(args.epochs):
        epoch_d_loss = 0
        epoch_g_adv_loss = 0
        epoch_g_rec_loss = 0
        epoch_gd_loss = 0
        epoch_gi_loss = 0
        start_time = time.time()

        for i, (clean_data, radial_length, identity_labels) in enumerate(train_loader):
            batch_size = clean_data.shape[0]

            # 将数据移至设备
            clean_data = clean_data.float().to(device)
            radial_length = radial_length.float().to(device)
            identity_labels = identity_labels.long().to(device)

            # 创建噪声数据
            noisy_data = add_noise_for_psnr(clean_data, psnr_level)

            # -----------------------
            # 1. 提取特征 - 使用当前特征提取器状态，但不立即更新
            # -----------------------
            with torch.no_grad():
                f_D, _ = G_D(clean_data)
                f_I, _ = G_I(clean_data)
                condition = torch.cat([f_D, f_I], dim=1)

            # -----------------------
            # 2. 训练判别器
            # -----------------------
            for _ in range(args.n_critic):  # 可以多次更新判别器
                optimizer_D.zero_grad()

                # 生成伪(去噪)样本
                with torch.no_grad():
                    generated_samples = generator(noisy_data, condition)

                # 创建标签，添加平滑化
                real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # 标签平滑化
                fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1  # 标签平滑化

                # 真实样本的判别器损失
                real_pred = discriminator(clean_data, condition)
                real_loss = adversarial_loss(real_pred, real_labels)

                # 伪样本的判别器损失
                fake_pred = discriminator(generated_samples.detach(), condition)
                fake_loss = adversarial_loss(fake_pred, fake_labels)

                # 总判别器损失
                d_loss = (real_loss + fake_loss) / 2

                # 添加梯度惩罚以提高稳定性（可选）
                if args.use_gp:
                    alpha = torch.rand(batch_size, 1).to(device)
                    interpolated = (alpha * clean_data + (1 - alpha) * generated_samples.detach()).requires_grad_(True)
                    interp_condition = condition.detach()  # 不计算条件的梯度

                    d_interp = discriminator(interpolated, interp_condition)

                    # 计算判别器输出相对于插值样本的梯度
                    gradients = torch.autograd.grad(
                        outputs=d_interp,
                        inputs=interpolated,
                        grad_outputs=torch.ones_like(d_interp),
                        create_graph=True,
                        retain_graph=True,
                    )[0]

                    gradients = gradients.view(batch_size, -1)
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                    d_loss = d_loss + args.lambda_gp * gradient_penalty

                d_loss.backward()

                # 梯度裁剪以防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip_value)

                optimizer_D.step()

            # -----------------------
            # 3. 训练生成器
            # -----------------------
            optimizer_G.zero_grad()

            # 重新生成样本
            generated_samples = generator(noisy_data, condition)

            # 生成器的对抗损失(欺骗判别器)
            g_adv_loss = adversarial_loss(discriminator(generated_samples, condition), real_labels)

            # 重建损失(生成和干净之间的MSE距离)
            g_rec_loss = reconstruction_loss(generated_samples, clean_data)

            # 总生成器损失
            g_loss = g_adv_loss + args.lambda_rec * g_rec_loss
            g_loss.backward()

            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_value)

            optimizer_G.step()

            # -----------------------
            # 4. 微调G_D（目标径向长度模块）
            # -----------------------
            optimizer_GD.zero_grad()

            # 使用干净数据进行径向长度预测
            _, pred_radial = G_D(clean_data)

            # 计算G_D的回归损失
            # 首先检查并处理可能的异常值
            valid_indices = ~torch.isnan(radial_length) & ~torch.isinf(radial_length) & (radial_length < 1e6)
            if valid_indices.sum() > 0:
                # 仅使用有效值计算损失
                gd_loss = regression_loss(
                    pred_radial[valid_indices],
                    radial_length[valid_indices]
                )

                # 确保损失不会过大
                if not torch.isnan(gd_loss) and not torch.isinf(gd_loss) and gd_loss < 1e6:
                    # 应用损失权重并反向传播
                    (args.lambda_gd * gd_loss).backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(G_D.parameters(), args.clip_value)
                    optimizer_GD.step()
                else:
                    print(f"跳过当前批次G_D更新：异常损失值 {gd_loss.item()}")
                    gd_loss = torch.tensor(0.0).to(device)
            else:
                print("跳过当前批次G_D更新：所有标签无效")
                gd_loss = torch.tensor(0.0).to(device)

            # -----------------------
            # 5. 微调G_I（目标身份模块）
            # -----------------------
            optimizer_GI.zero_grad()

            # 使用干净数据进行身份预测
            _, pred_identity = G_I(clean_data)

            # 计算G_I的分类损失
            gi_loss = classification_loss(pred_identity, identity_labels)

            # 应用损失权重并反向传播
            (args.lambda_gi * gi_loss).backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(G_I.parameters(), args.clip_value)

            optimizer_GI.step()

            # 更新epoch损失
            epoch_d_loss += d_loss.item()
            epoch_g_adv_loss += g_adv_loss.item()
            epoch_g_rec_loss += g_rec_loss.item()
            epoch_gd_loss += gd_loss.item()
            epoch_gi_loss += gi_loss.item()

            # 打印进度
            if i % 10 == 0:
                print(f"[Epoch {epoch + 1}/{args.epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G adv: {g_adv_loss.item():.4f}] "
                      f"[G rec: {g_rec_loss.item():.4f}] [G_D: {gd_loss.item():.4f}] "
                      f"[G_I: {gi_loss.item():.4f}]")

        # 计算该epoch的平均损失
        epoch_d_loss /= len(train_loader)
        epoch_g_adv_loss /= len(train_loader)
        epoch_g_rec_loss /= len(train_loader)
        epoch_gd_loss /= len(train_loader)
        epoch_gi_loss /= len(train_loader)

        # 添加到损失历史记录
        results["loss_history"]["discriminator"].append(epoch_d_loss)
        results["loss_history"]["generator_adv"].append(epoch_g_adv_loss)
        results["loss_history"]["generator_rec"].append(epoch_g_rec_loss)
        results["loss_history"]["G_D"].append(epoch_gd_loss)
        results["loss_history"]["G_I"].append(epoch_gi_loss)

        # 打印epoch摘要
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"D Loss: {epoch_d_loss:.4f}, G Adv Loss: {epoch_g_adv_loss:.4f}, "
              f"G Rec Loss: {epoch_g_rec_loss:.4f}, G_D Loss: {epoch_gd_loss:.4f}, "
              f"G_I Loss: {epoch_gi_loss:.4f} - Time: {epoch_time:.2f}s")

        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint_dir = os.path.join(psnr_save_dir, f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 保存所有模型
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, 'generator.pth'))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'discriminator.pth'))
            torch.save(G_D.state_dict(), os.path.join(checkpoint_dir, 'G_D.pth'))
            torch.save(G_I.state_dict(), os.path.join(checkpoint_dir, 'G_I.pth'))

            # 生成并保存样本去噪图像
            if args.save_samples:
                with torch.no_grad():
                    # 从数据集中获取样本
                    sample_idx = np.random.randint(0, len(train_dataset))
                    sample_clean, _, _ = train_dataset[sample_idx]
                    sample_clean = sample_clean.unsqueeze(0).float().to(device)

                    # 创建带噪声的样本
                    sample_noisy = add_noise_for_psnr(sample_clean, psnr_level)

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
                    plt.title(f'Noisy HRRP (PSNR={psnr_level}dB)')
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

    # 保存最终模型
    torch.save(generator.state_dict(), os.path.join(psnr_save_dir, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(psnr_save_dir, 'discriminator_final.pth'))
    torch.save(G_D.state_dict(), os.path.join(psnr_save_dir, 'G_D_final.pth'))
    torch.save(G_I.state_dict(), os.path.join(psnr_save_dir, 'G_I_final.pth'))

    results["model_paths"]["generator"] = os.path.join(psnr_save_dir, 'generator_final.pth')
    results["model_paths"]["discriminator"] = os.path.join(psnr_save_dir, 'discriminator_final.pth')
    results["model_paths"]["G_D"] = os.path.join(psnr_save_dir, 'G_D_final.pth')
    results["model_paths"]["G_I"] = os.path.join(psnr_save_dir, 'G_I_final.pth')

    # 绘制损失曲线
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(results["loss_history"]["discriminator"], label='Discriminator')
    plt.plot(results["loss_history"]["generator_adv"], label='Generator (Adv)')
    plt.plot(results["loss_history"]["generator_rec"], label='Generator (Rec)')
    plt.title('CGAN Adversarial Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(results["loss_history"]["G_D"], label='G_D')
    plt.plot(results["loss_history"]["G_I"], label='G_I')
    plt.title('Feature Extractors Fine-tuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(psnr_save_dir, 'training_loss.png'))
    plt.close()

    print(f"CGAN训练完成(PSNR={psnr_level}dB)，模型已保存至 {psnr_save_dir}")

    return results


def train_cae(args, device, psnr_level):
    """
    训练CAE用于HRRP去噪

    参数:
        args: 命令行参数
        device: 计算设备(CPU/GPU)
        psnr_level: 目标PSNR级别(dB)
    返回:
        字典，包含训练历史和模型路径
    """
    print(f"\n{'-' * 20} 开始训练CAE (PSNR={psnr_level}dB) {'-' * 20}")

    # 为当前PSNR级别创建保存目录
    psnr_save_dir = os.path.join(args.save_dir, f"cae_psnr_{psnr_level}dB")
    os.makedirs(psnr_save_dir, exist_ok=True)

    # 训练结果和历史记录
    results = {
        "loss_history": [],
        "model_paths": {}
    }

    # 创建CAE模型
    model = ConvAutoEncoder(input_dim=args.input_dim,
                            latent_dim=args.latent_dim,
                            hidden_dim=args.hidden_dim).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 加载数据集
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for i, (clean_data, _, _) in enumerate(train_loader):
            # 将数据移至设备
            clean_data = clean_data.float().to(device)

            # 创建噪声数据
            noisy_data = add_noise_for_psnr(clean_data, psnr_level)

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
        results["loss_history"].append(epoch_loss)

        # 打印epoch摘要
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{args.epochs} - Avg Loss: {epoch_loss:.4f} - Time: {epoch_time:.2f}s")

        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint_dir = os.path.join(psnr_save_dir, f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 保存模型
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'cae_model.pth'))

            # 保存样本去噪结果
            if args.save_samples:
                model.eval()
                with torch.no_grad():
                    # 从数据集中获取样本
                    sample_idx = np.random.randint(0, len(train_dataset))
                    sample_clean, _, _ = train_dataset[sample_idx]
                    sample_clean = sample_clean.unsqueeze(0).float().to(device)

                    # 创建带噪声的样本
                    sample_noisy = add_noise_for_psnr(sample_clean, psnr_level)

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
                    plt.title(f'Noisy HRRP (PSNR={psnr_level}dB)')
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
    final_model_path = os.path.join(psnr_save_dir, 'cae_model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    results["model_paths"]["cae"] = final_model_path

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(results["loss_history"], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CAE Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(psnr_save_dir, 'training_loss.png'))
    plt.close()

    print(f"CAE训练完成(PSNR={psnr_level}dB)，模型已保存至 {psnr_save_dir}")

    return results


def train_ae(args, device, psnr_level):
    """
    训练AE用于HRRP去噪

    参数:
        args: 命令行参数
        device: 计算设备(CPU/GPU)
        psnr_level: 目标PSNR级别(dB)
    返回:
        字典，包含训练历史和模型路径
    """
    print(f"\n{'-' * 20} 开始训练AE (PSNR={psnr_level}dB) {'-' * 20}")

    # 为当前PSNR级别创建保存目录
    psnr_save_dir = os.path.join(args.save_dir, f"ae_psnr_{psnr_level}dB")
    os.makedirs(psnr_save_dir, exist_ok=True)

    # 训练结果和历史记录
    results = {
        "loss_history": [],
        "model_paths": {}
    }

    # 创建AE模型
    model = AutoEncoder(input_dim=args.input_dim,
                        latent_dim=args.latent_dim,
                        hidden_dim=args.ae_hidden_dim).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 加载数据集
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for i, (clean_data, _, _) in enumerate(train_loader):
            # 将数据移至设备
            clean_data = clean_data.float().to(device)

            # 创建噪声数据
            noisy_data = add_noise_for_psnr(clean_data, psnr_level)

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
        results["loss_history"].append(epoch_loss)

        # 打印epoch摘要
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{args.epochs} - Avg Loss: {epoch_loss:.4f} - Time: {epoch_time:.2f}s")

        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint_dir = os.path.join(psnr_save_dir, f"epoch_{epoch + 1}")
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
                    sample_noisy = add_noise_for_psnr(sample_clean, psnr_level)

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
                    plt.title(f'Noisy HRRP (PSNR={psnr_level}dB)')
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
    final_model_path = os.path.join(psnr_save_dir, 'ae_model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    results["model_paths"]["ae"] = final_model_path

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(results["loss_history"], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('AE Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(psnr_save_dir, 'training_loss.png'))
    plt.close()

    print(f"AE训练完成(PSNR={psnr_level}dB)，模型已保存至 {psnr_save_dir}")

    return results


def main():
    """主程序入口，解析命令行参数并开始训练"""

    parser = argparse.ArgumentParser(description='HRRP去噪模型统一训练脚本')

    # 基本参数
    parser.add_argument('--model', type=str, default='all', choices=['modules', 'cgan', 'cae', 'ae', 'all'],
                        help='要训练的模型类型: modules-特征提取器，cgan-条件GAN，cae-卷积自编码器，ae-全连接自编码器，all-所有模型')
    parser.add_argument('--module', type=str, default='both', choices=['G_D', 'G_I', 'both'],
                        help='当model=modules时，要训练的特定模块: G_D-目标径向长度模块，G_I-目标身份模块，both-两者都训练')
    parser.add_argument('--train_dir', type=str, default='datasets/simulated_3/train',
                        help='包含训练数据的目录')
    parser.add_argument('--psnr_levels', type=float, nargs='+', default=[20, 10, 0],
                        help='要训练的PSNR级别列表(dB)')
    parser.add_argument('--seed', type=int, default=42,
                        help='可重现性的随机种子')

    # 模型保存参数
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='保存训练模型的根目录')
    parser.add_argument('--load_dir', type=str, default='checkpoints',
                        help='加载预训练特征提取器的目录')
    parser.add_argument('--load_feature_extractors', action='store_true',
                        help='是否加载预训练特征提取器')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='保存检查点的epoch间隔')
    parser.add_argument('--save_samples', action='store_true',
                        help='是否保存样本去噪结果')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='训练的批量大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练的epoch数')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率')

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

    # CGAN特定参数
    parser.add_argument('--lr_gd', type=float, default=0.00001,
                        help='G_D模块的学习率（应该很小以防止不稳定）')
    parser.add_argument('--lr_gi', type=float, default=0.0001,
                        help='G_I模块的学习率')
    parser.add_argument('--lambda_rec', type=float, default=10.0,
                        help='生成器损失中重建损失的权重')
    parser.add_argument('--lambda_gd', type=float, default=0.0001,
                        help='G_D回归损失的权重（大幅减小）')
    parser.add_argument('--lambda_gi', type=float, default=0.1,
                        help='G_I分类损失的权重')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='梯度惩罚项的权重')
    parser.add_argument('--n_critic', type=int, default=1,
                        help='每次更新生成器前更新判别器的次数')
    parser.add_argument('--use_gp', action='store_true',
                        help='是否使用梯度惩罚（WGAN-GP风格）')
    parser.add_argument('--clip_value', type=float, default=1.0,
                        help='梯度裁剪值')

    args = parser.parse_args()

    # 设置随机种子
    setup_seed(args.seed)

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 为每种模型创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练选择的模型
    start_time = time.time()

    if args.model == 'modules' or args.model == 'all':
        # 如果训练特征提取器模块，PSNR级别不适用
        modules_save_dir = os.path.join(args.save_dir, 'modules')
        os.makedirs(modules_save_dir, exist_ok=True)

        # 保存原始save_dir
        original_save_dir = args.save_dir
        args.save_dir = modules_save_dir

        # 训练特征提取器模块
        train_modules(args, device)

        # 恢复原始save_dir
        args.save_dir = original_save_dir

    # 根据所选模型和PSNR级别进行训练
    if args.model in ['cgan', 'cae', 'ae', 'all']:
        # 对每个PSNR级别训练指定的模型
        for psnr_level in args.psnr_levels:
            if args.model == 'cgan' or args.model == 'all':
                cgan_save_dir = os.path.join(args.save_dir, 'cgan')
                os.makedirs(cgan_save_dir, exist_ok=True)

                # 保存原始save_dir
                original_save_dir = args.save_dir
                args.save_dir = cgan_save_dir

                # 训练CGAN
                train_cgan(args, device, psnr_level)

                # 恢复原始save_dir
                args.save_dir = original_save_dir

            if args.model == 'cae' or args.model == 'all':
                cae_save_dir = os.path.join(args.save_dir, 'cae')
                os.makedirs(cae_save_dir, exist_ok=True)

                # 保存原始save_dir
                original_save_dir = args.save_dir
                args.save_dir = cae_save_dir

                # 训练CAE
                train_cae(args, device, psnr_level)

                # 恢复原始save_dir
                args.save_dir = original_save_dir

            if args.model == 'ae' or args.model == 'all':
                ae_save_dir = os.path.join(args.save_dir, 'ae')
                os.makedirs(ae_save_dir, exist_ok=True)

                # 保存原始save_dir
                original_save_dir = args.save_dir
                args.save_dir = ae_save_dir

                # 训练AE
                train_ae(args, device, psnr_level)

                # 恢复原始save_dir
                args.save_dir = original_save_dir

    # 计算并显示总训练时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n总训练时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")

    print(f"\n所有训练完成！模型已保存至 {args.save_dir}")


if __name__ == "__main__":
    main()