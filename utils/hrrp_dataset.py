from torch.utils.data import Dataset
from scipy.io import loadmat
import torch
import os
from torch.utils.data import DataLoader
import numpy as np
import re


class HRRPDataset(Dataset):
    def __init__(self, root_dir, dataset_type=None):
        super(HRRPDataset, self).__init__()
        self.root_dir = root_dir
        self.data_list = [f for f in os.listdir(root_dir) if f.endswith('.mat')]

        # Auto-detect dataset type if not specified
        if dataset_type is None:
            # Check a sample file name to determine dataset type
            if len(self.data_list) > 0:
                sample_file = self.data_list[0]
                if '_hrrp_measured_' in sample_file:
                    self.dataset_type = 'measured'
                else:
                    self.dataset_type = 'simulated'
            else:
                # Default to simulated if no files are found
                self.dataset_type = 'simulated'
        else:
            self.dataset_type = dataset_type

        print(f"Dataset type detected/set: {self.dataset_type}")

        # Create a mapping from target names to class indices
        target_names = set()
        for filename in self.data_list:
            if self.dataset_type == 'measured':
                # For measured data: an26_hrrp_measured_1.mat -> an26
                parts = filename.split('_')
                if len(parts) > 0:
                    target_name = parts[0]
                    target_names.add(target_name)
            else:
                # For simulated data: F15_hrrp_theta_75.0_phi_0.4_D_353.08.mat -> F15
                parts = filename.split('_')
                if len(parts) > 0:
                    target_name = parts[0]
                    target_names.add(target_name)

        self.target_to_idx = {name: idx for idx, name in enumerate(sorted(target_names))}
        print(f"Found {len(self.target_to_idx)} target classes: {', '.join(self.target_to_idx.keys())}")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path = os.path.join(self.root_dir, self.data_list[idx])
        if self.dataset_type == 'measured':
            data = abs(loadmat(data_path)['hrrp'].T)
        else:
            data = abs(loadmat(data_path)['CoHH'])
        # Handle different data shapes
        if len(data.shape) == 1:
            # Already 1D
            pass
        elif len(data.shape) == 2:
            # If 2D, flatten or take first row
            if data.shape[0] == 1:
                data = data.flatten()
            else:
                data = data[0, :]
        elif len(data.shape) == 3:
            # If 3D, take the first channel/slice
            data = data[0, 0, :]
        elif len(data.shape) == 4:
            # If 4D, take the first item
            data = data[0, 0, 0, :]

        # Convert to numpy array if not already
        data = np.array(data, dtype=np.float32)

        # Ensure 1D
        data = data.flatten()

        # Normalize data to [0, 1] range
        if np.max(data) > np.min(data):
            data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # Convert to tensor
        data = torch.from_numpy(data).float()

        # Extract radial length label from filename (if available)
        filename = self.data_list[idx]

        # Default radial length is 0.0 (especially for measured data where it's not available)
        radial_length = 0.0

        if self.dataset_type == 'simulated':
            # For simulated data: Extract D value
            match = re.search(r'_D_(\d+\.\d+)', filename)
            if match:
                try:
                    radial_length = float(match.group(1))
                except (ValueError, IndexError):
                    radial_length = 0.0

        # Extract target identity from filename
        if self.dataset_type == 'measured':
            # For measured data: an26_hrrp_measured_1.mat -> an26
            parts = filename.split('_')
            target_name = parts[0] if len(parts) > 0 else "unknown"
        else:
            # For simulated data: F15_hrrp_theta_75.0_phi_0.4_D_353.08.mat -> F15
            parts = filename.split('_')
            target_name = parts[0] if len(parts) > 0 else "unknown"

        identity_label = self.target_to_idx.get(target_name, 0)

        return data, radial_length, identity_label

    def __len__(self):
        return len(self.data_list)

    def get_num_classes(self):
        return len(self.target_to_idx)

    def get_dataset_type(self):
        return self.dataset_type


if __name__ == "__main__":
    # Test with simulated data
    print("Testing with simulated data:")
    dataset = HRRPDataset("../datasets/simulated_3/train")
    for i in range(min(5, len(dataset))):  # Print info for the first 5 samples
        data, radial_length, identity = dataset[i]
        print(f"Sample {i}:")
        print(f"  HRRP Shape: {data.shape}")
        print(f"  Radial Length: {radial_length}")
        print(f"  Identity Class: {identity}")

    # Test with measured data (if available)
    try:
        print("\nTesting with measured data:")
        dataset = HRRPDataset("../datasets/measured_3/train")
        for i in range(min(5, len(dataset))):  # Print info for the first 5 samples
            data, radial_length, identity = dataset[i]
            print(f"Sample {i}:")
            print(f"  HRRP Shape: {data.shape}")
            print(f"  Radial Length: {radial_length}")
            print(f"  Identity Class: {identity}")
    except:
        print("Measured dataset not available for testing")