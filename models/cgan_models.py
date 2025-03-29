import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generator model for CGAN, transforming noisy HRRP data into clean HRRP data.
    Using 1D convolutions for better signal processing.
    Supports both simulated data (with radial length features) and measured data (without).
    """

    def __init__(self, input_dim=500, condition_dim=128, hidden_dim=128, dataset_type='simulated'):
        """
        Parameters:
            input_dim (int): Dimension of input noisy HRRP sequence
            condition_dim (int): Dimension of condition vector (identity+radial features)
            hidden_dim (int): Dimension of hidden layers
            dataset_type (str): 'simulated' or 'measured' - affects conditioning mechanism
        """
        super(Generator, self).__init__()

        self.dataset_type = dataset_type

        # For measured data without radial length, we might have a smaller condition vector
        effective_condition_dim = condition_dim
        if dataset_type == 'measured':
            # If we're not using radial length features, condition_dim might be halved
            # This is handled by the caller - we just use whatever condition_dim is provided
            pass

        # Embedding layer for condition
        self.condition_fc = nn.Linear(effective_condition_dim, input_dim)

        # Convolutional layers for processing combined input
        self.conv1 = nn.Conv1d(2, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv1d(hidden_dim, 1, kernel_size=5, stride=1, padding=2)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, condition):
        """
        Forward pass of the generator

        Parameters:
            x (torch.Tensor): Input noisy HRRP sequence, shape [batch_size, input_dim]
            condition (torch.Tensor): Condition tensor, shape [batch_size, condition_dim]
                For simulated data: Combined identity and radial length features
                For measured data: Identity features only

        Returns:
            torch.Tensor: Generated clean HRRP, shape [batch_size, input_dim]
        """
        batch_size = x.size(0)

        # Reshape input to [batch_size, 1, input_dim] for 1D convolution
        x = x.unsqueeze(1)

        # Process condition and reshape to [batch_size, 1, input_dim]
        condition = self.condition_fc(condition)
        condition = condition.unsqueeze(1)

        # Concatenate input and condition along channel dimension
        x = torch.cat([x, condition], dim=1)  # [batch_size, 2, input_dim]

        # Apply convolutional layers with batch normalization and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))

        # Flatten output to [batch_size, input_dim]
        x = x.view(batch_size, -1)

        return x


class Discriminator(nn.Module):
    """
    Discriminator model for CGAN, distinguishing between real clean HRRP data and generated clean HRRP data.
    Using 1D convolutions for better signal processing.
    Supports both simulated data (with radial length features) and measured data (without).
    """

    def __init__(self, input_dim=500, condition_dim=128, hidden_dim=128, dataset_type='simulated'):
        """
        Parameters:
            input_dim (int): Dimension of input HRRP sequence (clean or generated)
            condition_dim (int): Dimension of condition vector (identity+radial features)
            hidden_dim (int): Dimension of hidden layers
            dataset_type (str): 'simulated' or 'measured' - affects conditioning mechanism
        """
        super(Discriminator, self).__init__()

        self.dataset_type = dataset_type

        # For measured data without radial length, we might have a smaller condition vector
        effective_condition_dim = condition_dim
        if dataset_type == 'measured':
            # If we're not using radial length features, condition_dim might be halved
            # This is handled by the caller - we just use whatever condition_dim is provided
            pass

        # Embedding layer for condition
        self.condition_fc = nn.Linear(effective_condition_dim, input_dim)

        # Convolutional layers
        self.conv1 = nn.Conv1d(2, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=5, stride=2, padding=2)

        # Calculate flattened size after convolutions
        self.flattened_size = hidden_dim * 4 * ((input_dim + 7) // 8)  # Round up to handle non-divisible sizes

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, condition):
        """
        Forward pass of the discriminator

        Parameters:
            x (torch.Tensor): Input HRRP sequence (clean or generated), shape [batch_size, input_dim]
            condition (torch.Tensor): Condition tensor, shape [batch_size, condition_dim]
                For simulated data: Combined identity and radial length features
                For measured data: Identity features only

        Returns:
            torch.Tensor: Discrimination result, shape [batch_size, 1]
        """
        batch_size = x.size(0)

        # Reshape input to [batch_size, 1, input_dim] for 1D convolution
        x = x.unsqueeze(1)

        # Process condition and reshape to [batch_size, 1, input_dim]
        condition = self.condition_fc(condition)
        condition = condition.unsqueeze(1)

        # Concatenate input and condition along channel dimension
        x = torch.cat([x, condition], dim=1)  # [batch_size, 2, input_dim]

        # Apply convolutional layers with LeakyReLU
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))

        # Flatten
        x = x.view(batch_size, -1)

        # Apply fully connected layers
        x = self.leaky_relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x