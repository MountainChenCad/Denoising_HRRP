import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    用于CGAN的生成器模型，将噪声HRRP数据生成为干净HRRP数据。
    使用1D卷积进行更好的信号处理。
    """

    def __init__(self, input_dim=500, condition_dim=128, hidden_dim=128):
        """
        参数:
            input_dim (int): 输入噪声HRRP序列的维度
            condition_dim (int): 条件向量的维度(身份+径向特征)
            hidden_dim (int): 隐藏层的维度
        """
        super(Generator, self).__init__()

        # 条件的嵌入层
        self.condition_fc = nn.Linear(condition_dim, input_dim)

        # 处理组合输入的卷积层
        self.conv1 = nn.Conv1d(2, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv1d(hidden_dim, 1, kernel_size=5, stride=1, padding=2)

        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, condition):
        """
        生成器的前向传播

        参数:
            x (torch.Tensor): 输入噪声HRRP序列，形状为 [batch_size, input_dim]
            condition (torch.Tensor): 条件张量，形状为 [batch_size, condition_dim]

        返回:
            torch.Tensor: 生成的干净HRRP，形状为 [batch_size, input_dim]
        """
        batch_size = x.size(0)

        # 将输入重塑为 [batch_size, 1, input_dim] 用于1D卷积
        x = x.unsqueeze(1)

        # 处理条件并重塑为 [batch_size, 1, input_dim]
        condition = self.condition_fc(condition)
        condition = condition.unsqueeze(1)

        # 沿通道维度连接输入和条件
        x = torch.cat([x, condition], dim=1)  # [batch_size, 2, input_dim]

        # 应用带有批归一化和ReLU的卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))

        # 将输出展平为 [batch_size, input_dim]
        x = x.view(batch_size, -1)

        return x


class Discriminator(nn.Module):
    """
    用于CGAN的判别器模型，区分真实干净HRRP数据和生成的干净HRRP数据。
    使用1D卷积进行更好的信号处理。
    """

    def __init__(self, input_dim=500, condition_dim=128, hidden_dim=128):
        """
        参数:
            input_dim (int): 输入HRRP序列的维度(干净或生成的)
            condition_dim (int): 条件向量的维度(身份+径向特征)
            hidden_dim (int): 隐藏层的维度
        """
        super(Discriminator, self).__init__()

        # 条件的嵌入层
        self.condition_fc = nn.Linear(condition_dim, input_dim)

        # 卷积层
        self.conv1 = nn.Conv1d(2, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=5, stride=2, padding=2)

        # 计算卷积后展平的大小
        self.flattened_size = hidden_dim * 4 * ((input_dim + 7) // 8)  # 向上取整以处理不可整除的大小

        # 全连接层
        self.fc1 = nn.Linear(self.flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Leaky ReLU激活
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, condition):
        """
        判别器的前向传播

        参数:
            x (torch.Tensor): 输入HRRP序列(干净或生成的)，形状为 [batch_size, input_dim]
            condition (torch.Tensor): 条件张量，形状为 [batch_size, condition_dim]

        返回:
            torch.Tensor: 判别结果，形状为 [batch_size, 1]
        """
        batch_size = x.size(0)

        # 将输入重塑为 [batch_size, 1, input_dim] 用于1D卷积
        x = x.unsqueeze(1)

        # 处理条件并重塑为 [batch_size, 1, input_dim]
        condition = self.condition_fc(condition)
        condition = condition.unsqueeze(1)

        # 沿通道维度连接输入和条件
        x = torch.cat([x, condition], dim=1)  # [batch_size, 2, input_dim]

        # 应用带有LeakyReLU的卷积层
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))

        # 展平
        x = x.view(batch_size, -1)

        # 应用全连接层
        x = self.leaky_relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x