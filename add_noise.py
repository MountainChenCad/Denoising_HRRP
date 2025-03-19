import os
import numpy as np
from scipy.io import loadmat, savemat
from tqdm import tqdm
import torch


def add_noise_for_psnr(signal, psnr_db):
    """
    添加高斯噪声以达到特定的PSNR(峰值信噪比)值

    Args:
        signal (numpy.ndarray): 原始干净信号
        psnr_db (float): 期望的PSNR值(dB)

    Returns:
        numpy.ndarray: 添加噪声后的信号
    """
    # 计算信号功率
    signal_power = np.mean(signal ** 2)

    # 根据期望的PSNR计算噪声功率
    noise_power = signal_power / (10 ** (psnr_db / 10))

    # 生成高斯噪声
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    # 将噪声添加到信号中
    noisy_signal = signal + noise

    return noisy_signal


def create_noisy_dataset(src_dir, dest_base_dir, psnr_levels=[0, 10, 20]):
    """
    从原始数据创建具有指定PSNR级别的噪声数据集

    Args:
        src_dir (str): 包含原始HRRP数据的目录
        dest_base_dir (str): 存储噪声数据集的基本目录
        psnr_levels (list): 要生成的PSNR级别列表(dB)
    """
    # 获取源目录中的所有.mat文件
    files = [f for f in os.listdir(src_dir) if f.endswith('.mat')]

    # 为每个PSNR级别创建目标目录
    for psnr in psnr_levels:
        dest_dir = os.path.join(dest_base_dir, f"psnr_{psnr}")
        os.makedirs(dest_dir, exist_ok=True)

    # 处理每个文件
    for file in tqdm(files, desc="处理文件"):
        file_path = os.path.join(src_dir, file)

        # 加载数据
        data = loadmat(file_path)
        signal = abs(data['CoHH'])

        # 对每个PSNR级别
        for psnr in psnr_levels:
            # 添加噪声以达到指定的PSNR
            noisy_signal = add_noise_for_psnr(signal, psnr)

            # 保存带噪声的信号
            dest_file = os.path.join(dest_base_dir, f"psnr_{psnr}", file)
            data_noisy = data.copy()
            data_noisy['CoHH'] = noisy_signal
            savemat(dest_file, data_noisy)


def main():
    """主函数，处理训练集和测试集"""
    # 设置随机种子以便结果可重现
    np.random.seed(42)
    torch.manual_seed(42)

    # 定义源目录和目标目录
    train_src_dir = "datasets/simulated_3/train"
    test_src_dir = "datasets/simulated_3/test"

    train_dest_base_dir = "datasets/simulated_3_noisy/train"
    test_dest_base_dir = "datasets/simulated_3_noisy/test"

    # 创建基本目录
    os.makedirs(train_dest_base_dir, exist_ok=True)
    os.makedirs(test_dest_base_dir, exist_ok=True)

    # 要生成的PSNR级别
    psnr_levels = [0, 10, 20]

    # 创建带噪声的训练数据集
    print("创建带噪声的训练数据集...")
    create_noisy_dataset(train_src_dir, train_dest_base_dir, psnr_levels)

    # 创建带噪声的测试数据集
    print("创建带噪声的测试数据集...")
    create_noisy_dataset(test_src_dir, test_dest_base_dir, psnr_levels)

    print("完成!")


if __name__ == "__main__":
    main()