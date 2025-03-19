import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from models import TargetRadialLengthModule, TargetIdentityModule
from hrrp_dataset import HRRPDataset


def train_G_D(args):
    """
    Train the Target Radial Length Module (G_D)

    Args:
        args: Training arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load dataset
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Print some info about the dataset
    sample_data, sample_radial, _ = next(iter(train_loader))
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample radial length: {sample_radial.shape}")

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (data, radial_length, _) in enumerate(train_loader):
            # Move data to device
            data = data.float().to(device)
            radial_length = radial_length.float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            _, predicted_radial_length = model(data)

            # Calculate loss
            loss = criterion(predicted_radial_length, radial_length)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.4f}")
                running_loss = 0.0

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'G_D.pth'))
    print(f"Target Radial Length Module (G_D) saved to {os.path.join(args.save_dir, 'G_D.pth')}")


def train_G_I(args):
    """
    Train the Target Identity Module (G_I)

    Args:
        args: Training arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                                 num_classes=args.num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load dataset
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Print some info about the dataset
    sample_data, _, sample_identity = next(iter(train_loader))
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample identity: {sample_identity.shape}")
    print(f"Number of classes: {train_dataset.get_num_classes()}")

    # Update num_classes based on dataset
    args.num_classes = train_dataset.get_num_classes()
    model = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                                 num_classes=args.num_classes).to(device)

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (data, _, identity_labels) in enumerate(train_loader):
            # Move data to device
            data = data.float().to(device)
            # Convert labels to integers for CrossEntropyLoss
            identity_labels = identity_labels.long().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            _, identity_logits = model(data)

            # Calculate loss
            loss = criterion(identity_logits, identity_labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(identity_logits.data, 1)
            total += identity_labels.size(0)
            correct += (predicted == identity_labels).sum().item()

            if i % 10 == 9:  # Print every 10 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.4f}, accuracy: {100 * correct / total:.2f}%")
                running_loss = 0.0
                correct = 0
                total = 0

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'G_I.pth'))
    print(f"Target Identity Module (G_I) saved to {os.path.join(args.save_dir, 'G_I.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train HRRP Modules')
    parser.add_argument('--module', type=str, default='both', choices=['G_D', 'G_I', 'both'],
                        help='Module to train (G_D, G_I, or both)')
    parser.add_argument('--train_dir', type=str, default='datasets/simulated_3/train',
                        help='Directory containing training data')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save trained models')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--input_dim', type=int, default=500,
                        help='Dimension of input HRRP sequence')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Dimension of output feature')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of target identity classes')

    args = parser.parse_args()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Train modules
    if args.module in ['G_D', 'both']:
        print("Training Target Radial Length Module (G_D)...")
        train_G_D(args)

    if args.module in ['G_I', 'both']:
        print("Training Target Identity Module (G_I)...")
        train_G_I(args)