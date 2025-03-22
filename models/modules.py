import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetRadialLengthModule(nn.Module):
    """
    G_D: Target Radial Length Module
    A simple 1-D CNN with three layers to extract target radial length information
    """

    def __init__(self, input_dim=500, feature_dim=64):
        """
        Args:
            input_dim (int): Dimension of input HRRP sequence
            feature_dim (int): Dimension of output feature
        """
        super(TargetRadialLengthModule, self).__init__()

        # Layer 1: Conv1d with ReLU activation
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)

        # Layer 2: Conv1d with ReLU activation
        self.conv2 = nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1)

        # Calculate flattened size after convolutions
        self.flatten_size = input_dim

        # Layer 3: Fully connected layer
        self.fc = nn.Linear(self.flatten_size, feature_dim)

    def forward(self, x):
        """
        Forward pass of the G_D module

        Args:
            x (torch.Tensor): Input HRRP sequence with shape [batch_size, input_dim]

        Returns:
            torch.Tensor: High-dimensional feature f_D with shape [batch_size, feature_dim]
            torch.Tensor: Predicted radial length (for training)
        """
        # Ensure the input is 2D [batch_size, input_dim]
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        # Reshape input to [batch_size, 1, input_dim] for 1D convolution
        x = x.unsqueeze(1)

        # Apply first convolutional layer with ReLU
        x = F.relu(self.conv1(x))

        # Apply second convolutional layer with ReLU
        x = F.relu(self.conv2(x))

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Apply fully connected layer to get feature representation
        features = self.fc(x)

        # Return both features and radial length prediction
        radial_length = torch.mean(features, dim=1)  # Simple prediction mechanism

        return features, radial_length


class TargetIdentityModule(nn.Module):
    """
    G_I: Target Identity Module
    A simple 1-D CNN with four layers to extract target identity information
    """

    def __init__(self, input_dim=500, feature_dim=64, num_classes=3):
        """
        Args:
            input_dim (int): Dimension of input HRRP sequence
            feature_dim (int): Dimension of output feature
            num_classes (int): Number of target identity classes
        """
        super(TargetIdentityModule, self).__init__()

        # Layer 1: Conv1d with ReLU activation
        self.conv1 = nn.Conv1d(1, 512, kernel_size=3, stride=1, padding=1)

        # Layer 2: Maxpool with ReLU activation
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Calculate flattened size after convolutions and pooling
        self.flatten_size = 512 * (input_dim // 2)

        # Layer 3: Fully connected layer with LeakyReLU activation
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.leaky_relu = nn.LeakyReLU(0.01)

        # Layer 4: Fully connected layer for identity classification
        self.fc_identity = nn.Linear(256, num_classes)

        # Feature layer: For generating features to be used by the fusion module
        self.fc_feature = nn.Linear(256, feature_dim)

    def forward(self, x):
        """
        Forward pass of the G_I module

        Args:
            x (torch.Tensor): Input HRRP sequence with shape [batch_size, input_dim]

        Returns:
            torch.Tensor: High-dimensional feature f_I with shape [batch_size, feature_dim]
            torch.Tensor: Predicted identity class logits (for training)
        """
        # Ensure the input is 2D [batch_size, input_dim]
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        # Reshape input to [batch_size, 1, input_dim] for 1D convolution
        x = x.unsqueeze(1)

        # Apply first convolutional layer with ReLU
        x = F.relu(self.conv1(x))

        # Apply maxpool layer with ReLU
        x = F.relu(self.pool(x))

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Apply first fully connected layer with LeakyReLU
        x = self.leaky_relu(self.fc1(x))

        # Get identity prediction logits
        identity_logits = self.fc_identity(x)

        # Get feature representation
        features = self.fc_feature(x)

        return features, identity_logits