from torch.utils.data import Dataset
from scipy.io import loadmat
import torch
import os
from torch.utils.data import DataLoader
import numpy as np


class HRRPDataset(Dataset):
    def __init__(self, root_dir):
        super(HRRPDataset, self).__init__()
        self.root_dir = root_dir
        self.data_list = [f for f in os.listdir(root_dir) if f.endswith('.mat')]

        # Create a mapping from target names to class indices
        # This assumes that target names are in the filename as described
        target_names = set()
        for filename in self.data_list:
            parts = filename.split('_')
            if len(parts) > 0:
                target_name = parts[0]  # Assuming target name is the first part of the filename
                target_names.add(target_name)

        self.target_to_idx = {name: idx for idx, name in enumerate(sorted(target_names))}

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path = os.path.join(self.root_dir, self.data_list[idx])
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
        parts = filename.split('_')
        radial_length = 0.0
        if len(parts) > 7 and parts[7].endswith('.mat'):
            try:
                label_str = parts[7][:-4]  # Remove .mat extension
                radial_length = float(label_str)
            except (ValueError, IndexError):
                radial_length = 0.0

        # Extract target identity from filename
        target_name = parts[0] if len(parts) > 0 else "unknown"
        identity_label = self.target_to_idx.get(target_name, 0)

        return data, radial_length, identity_label

    def __len__(self):
        return len(self.data_list)

    def get_num_classes(self):
        return len(self.target_to_idx)


if __name__ == "__main__":
    dataset = HRRPDataset("datasets/simulated_3/train")
    for i in range(min(5, len(dataset))):  # Print info for the first 5 samples
        data, radial_length, identity = dataset[i]
        print(f"Sample {i}:")
        print(f"  HRRP Shape: {data.shape}")
        print(f"  Radial Length: {radial_length}")
        print(f"  Identity Class: {identity}")