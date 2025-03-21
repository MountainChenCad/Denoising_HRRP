# ae_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """
    基于全连接层的自编码器(AutoEncoder)用于HRRP信号去噪。
    使用线性层进行编码和解码，相比CAE更简单但仍然有效。
    """

    def __init__(self, input_dim=500, latent_dim=64, hidden_dim=256):
        """
        参数:
            input_dim (int): 输入HRRP序列的维度
            latent_dim (int): 潜在空间表示的维度
            hidden_dim (int): 隐藏层的维度
        """
        super(AutoEncoder, self).__init__()

        # 存储维度
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(True)
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出范围为 [0, 1]
        )

    def encode(self, x):
        """
        将HRRP数据编码为潜在表示

        参数:
            x (torch.Tensor): 输入HRRP数据 [batch_size, input_dim]

        返回:
            torch.Tensor: 潜在表示 [batch_size, latent_dim]
        """
        return self.encoder(x)

    def decode(self, latent):
        """
        从潜在表示解码为重建的HRRP

        参数:
            latent (torch.Tensor): 潜在表示 [batch_size, latent_dim]

        返回:
            torch.Tensor: 重建的HRRP数据 [batch_size, input_dim]
        """
        return self.decoder(latent)

    def forward(self, x):
        """
        自编码器的前向传播

        参数:
            x (torch.Tensor): 输入HRRP数据 [batch_size, input_dim]

        返回:
            torch.Tensor: 重建的HRRP数据 [batch_size, input_dim]
            torch.Tensor: 潜在表示 [batch_size, latent_dim]
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent