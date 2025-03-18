from torch.utils.data import Dataset
from scipy.io import loadmat
import torch
import os
from torch.utils.data import DataLoader
class HRRPDataset(Dataset):
    def __init__(self, root_dir):
        super(HRRPDataset, self).__init__()
        self.root_dir = root_dir
        self.data_list = [f for f in os.listdir(root_dir) if f.endswith('.mat')]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path = os.path.join(self.root_dir, self.data_list[idx])
        data =  abs(loadmat(data_path)['CoHH'])

        # 从文件名中提取标签（径向长度）
        label_str = self.data_list[idx].split('_')[7]
        label = float(label_str[:-4])

        return data, label


    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":
    dataset = HRRPDataset("train")
    for i in range(5):  # 打印前5个样本的信息
        sample = dataset[i]
        print(f"Sample {i}: Vector Shape - {sample[0].shape}, Label - {sample[1]}")