import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as io


def open_mat(path):
    # Open .mat file
    dic_file = io.loadmat(path)

    # Extract fields and convert them in float32 numpy arrays
    pan_np = dic_file['I_PAN'].astype(np.float32)
    ms_lr_np = dic_file['I_MS_LR'].astype(np.float32)
    ms_np = dic_file['I_MS'].astype(np.float32)

    if 'I_GT' in dic_file.keys():
        gt_np = dic_file['I_GT'].astype(np.float32)
        gt = torch.from_numpy(np.moveaxis(gt_np, -1, 0)[None, :, :, :])
    else:
        gt = None

    # Convert numpy arrays to torch tensors
    ms_lr = torch.from_numpy(np.moveaxis(ms_lr_np, -1, 0)[None, :, :, :])
    pan = torch.from_numpy(pan_np[None, None, :, :])
    ms = torch.from_numpy(np.moveaxis(ms_np, -1, 0)[None, :, :, :])
    wavelenghts = torch.from_numpy(dic_file['Wavelengths']).float()

    return pan, ms_lr, ms, gt, wavelenghts


class TrainingDatasetFR(Dataset):
    def __init__(self, img_paths, norm):
        super(TrainingDatasetFR, self).__init__()

        pan = []
        ms_lr = []
        ms = []

        for i in range(len(img_paths)):
            pan_single, ms_lr_single, ms_single, _, _ = open_mat(img_paths[i])
            pan.append(pan_single.float())
            ms_lr.append(ms_lr_single.float())
            ms.append(ms_single.float())

        pan = torch.cat(pan, 0)
        ms_lr = torch.cat(ms_lr, 0)
        ms = torch.cat(ms, 0)

        pan = norm(pan)
        ms_lr = norm(ms_lr)
        ms = norm(ms)

        self.pan = pan
        self.ms_lr = ms_lr
        self.ms = ms

    def __len__(self):
        return self.pan.shape[0]

    def __getitem__(self, index):
        return self.pan[index], self.ms_lr[index], self.ms[index]
