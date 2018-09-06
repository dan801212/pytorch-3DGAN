from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import ReadVoxLabel
import torch
import scipy.io as io
import os.path
import scipy.ndimage as nd


class CustomDataset(Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root):
        """Set the path for Data.
        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)

    def __getitem__(self, index):
        with open(self.root + self.listdir[index], "rb") as f:
            volume = np.asarray(getVoxelFromMat(f, 64), dtype=np.float32)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)


def getVoxelFromMat(path, cube_len=64):
	#if we use shapenet dataset, we have to upsample the data. 32 -> 64
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels