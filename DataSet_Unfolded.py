import torch
import numpy as np
import torch.utils.data as data
from scipy.io import loadmat


def preprocess(L, S, D):
    A = max(np.max(np.abs(L)),np.max(np.abs(S)),np.max(np.abs(D)))
    if A == 0:
        A = 1
    L = L/A
    S = S/A
    D = D/A
    return L,S,D
    
class ImageDataset(data.Dataset):

    DATA_DIR='./Data/'
    def __init__(self, shape, train, transform=None, data_dir=None, length=10):
        self.DATA_DIR = data_dir if data_dir is None else data_dir
        self.shape = shape
        self.train = train
        self.length = length
        self.transform = transform

    def __getitem__(self, index):
        assert index < self.length

        if self.train is 0:

            L = np.load(self.DATA_DIR + 'processedData/train/L_xca%.5d.npy' % (index))
            S = np.load(self.DATA_DIR + 'processedData/train/S_xca%.5d.npy' % (index))
            D = np.load(self.DATA_DIR + 'processedData/train/D_xca%.5d.npy' % (index))
            L, S, D = preprocess(L, S, D)

            L_tensor = torch.from_numpy(L.reshape(self.shape))
            S_tensor = torch.from_numpy(S.reshape(self.shape))
            D_tensor = torch.from_numpy(D.reshape(self.shape))

        #   --  VALIDATION -- RAT 2, 100 frames
        if self.train is 1:

            L = np.load(self.DATA_DIR + 'processedData/val/L_xca%.5d.npy' % (index))
            S = np.load(self.DATA_DIR + 'processedData/val/S_xca%.5d.npy' % (index))
            D = np.load(self.DATA_DIR + 'processedData/val/D_xca%.5d.npy' % (index))
            L, S, D = preprocess(L, S, D)

            L_tensor = torch.from_numpy(L.reshape(self.shape))
            S_tensor = torch.from_numpy(S.reshape(self.shape))
            D_tensor = torch.from_numpy(D.reshape(self.shape))

        if self.train is 2:

            L = np.load(self.DATA_DIR + 'processedData/test/L_xca%.5d.npy' % (index))
            S = np.load(self.DATA_DIR + 'processedData/test/S_xca%.5d.npy' % (index))
            D = np.load(self.DATA_DIR + 'processedData/test/D_xca%.5d.npy' % (index))
            L, S, D = preprocess(L, S, D)

            L_tensor = torch.from_numpy(L.reshape(self.shape))
            S_tensor = torch.from_numpy(S.reshape(self.shape))
            D_tensor = torch.from_numpy(D.reshape(self.shape))


        return L_tensor, S_tensor, D_tensor

    def __len__(self):
        return self.length

