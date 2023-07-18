import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path)
        self.measurements = torch.tensor(self.data["measurements"], dtype=torch.float32)
        self.trajectories = torch.tensor(self.data["trajectories"], dtype=torch.float32)

    def __len__(self):
        return self.measurements.shape[0]

    def __getitem__(self, index):
        return {
            "measurements": self.measurements[index],
            "trajectories": self.trajectories[index],
        }
