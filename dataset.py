from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import torch
from modules.utils import timer

class PatchDataset(Dataset):
    def __init__(self, path, isTrain=False):
        """
        path: Path to folder containing folders train and test with each folder containing videos
        """
        if isTrain:
            self.path = os.path.join(path, "train")
        else:
            self.path = os.path.join(path, "test")

        with open(os.path.join(self.path, "indices.pkl"), 'rb') as file:
            self.indices = pickle.load(file)
        
        self.length = sorted(self.indices.keys())[-1] + 1   
        
        with open(os.path.join(self.path, "indices.pkl"), 'rb') as file:
            self.indices = pickle.load(file)

        self.fileindex = -1
        self.patches_src = None
        self.patches_dst = None

    def __len__(self):
       return self.length
    
    # @timer
    def __getitem__(self, idx):
        index_info = self.indices[idx]
        file_index = index_info[0]
        if(self.fileindex != file_index):
            self.patches_src = np.load(os.path.join(self.path, "src", "{}.npy".format(file_index)))
            self.patches_dst = np.load(os.path.join(self.path, "dst", "{}.npy".format(file_index)))
            self.fileindex = file_index

        patch_index = idx - index_info[1]

        src = torch.from_numpy(self.patches_src[patch_index, :]).type(torch.float32)
        dst = torch.from_numpy(self.patches_dst[patch_index, :]).type(torch.float32)

        return (src, dst)



