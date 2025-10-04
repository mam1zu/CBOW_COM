import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #data = {'centre': self.data[idx][0], 'left': self.data[idx][1: self.window_size], 'right': self.data[idx][self.window_size+1 : self.window_size * 2]}
        sample = {'data': self.data[idx], 'labels': self.labels[idx]}
        return sample
    