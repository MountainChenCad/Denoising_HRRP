import torch
from models.modules import TargetRadialLengthModule, TargetIdentityModule
import matplotlib.pyplot as plt
import argparse
import os
from utils.hrrp_dataset import HRRPDataset
from torch.utils.data import DataLoader


def test_G_D(args):
    """
    Test the Target Radial Length Module (G_D)

    Args:
        args: Testing arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(os.path.join(args.load_dir, 'G_D.pth')))
    model.eval()

    # Load dataset
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Test a few samples
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # Move data to device
            data = data.float().to(device)
            label = label.float().to(device)

            # Forward pass
            features, predicted_radial_length = model(data)

            # Print results
            print(f"Sample {i + 1}:")
            print(f"  Actual Radial Length: {label.item():.4f}")
            print(f"  Predicted Radial Length: {predicted_radial_length.item():.4f}")
            print(f"  Feature Dimensionality: {features.shape}")

            # Plot HRRP signal
            plt.figure(figsize=(10, 4))
            plt.plot(data.cpu().numpy()[0])
            plt.title(f"HRRP Signal (Radial Length: {label.item():.4f})")
            plt.xlabel("Range Bin")
            plt.ylabel("Amplitude")
            plt.savefig(f"sample_{i + 1}_G_D.png")
            plt.close()


def test_G_I(args):
    """
    Test the Target Identity Module (G_I)

    Args:
        args: Testing arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                                 num_classes=args.num_classes).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(os.path.join(args.load_dir, 'G_I.pth')))
    model.eval()

    # Load dataset
    test_dataset = HRRPDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Test a few samples
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # Move data to device
            data = data.float().to(device)

            # Forward pass
            features, identity_logits = model(data)

            # Convert logits to probabilities and get predicted class
            probabilities = torch.softmax(identity_logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

            # Print results
            print(f"Sample {i + 1}:")
            print(f"  Actual Identity Class: {int(label.item())}")
            print(f"  Predicted Identity Class: {predicted_class}")
            print(f"  Class Probabilities: {probabilities.cpu().numpy()[0]}")
            print(f"  Feature Dimensionality: {features.shape}")

            # Plot HRRP signal
            plt.figure(figsize=(10, 4))
            plt.plot(data.cpu().numpy()[0])
            plt.title(f"HRRP Signal (Identity Class: {int(label.item())})")
            plt.xlabel("Range Bin")
            plt.ylabel("Amplitude")
            plt.savefig(f"sample_{i + 1}_G_I.png")
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test HRRP Modules')
    parser.add_argument('--module', type=str, default='both', choices=['G_D', 'G_I', 'both'],
                        help='Module to test (G_D, G_I, or both)')
    parser.add_argument('--test_dir', type=str, default='dataset/test',
                        help='Directory containing test data')
    parser.add_argument('--load_dir', type=str, default='checkpoints',
                        help='Directory to load trained models from')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of test samples to process')
    parser.add_argument('--input_dim', type=int, default=256,
                        help='Dimension of input HRRP sequence')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Dimension of output feature')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of target identity classes')

    args = parser.parse_args()

    # Test modules
    if args.module in ['G_D', 'both']:
        print("Testing Target Radial Length Module (G_D)...")
        test_G_D(args)

    if args.module in ['G_I', 'both']:
        print("Testing Target Identity Module (G_I)...")
        test_G_I(args)