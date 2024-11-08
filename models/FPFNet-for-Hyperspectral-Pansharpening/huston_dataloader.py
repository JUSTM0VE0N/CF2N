from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import os



class My_dataset_huston(Dataset):
    def __init__(self, root, mode):
        super(My_dataset_huston, self).__init__()
        self.root = root
        self.mode = mode
        self.gtHS = []
        self.LRHS = []
        self.PAN = []
        if self.mode == "train":
            self.gtHS = os.listdir(os.path.join(self.root, "train", "gtHS"))
            self.gtHS.sort(key=lambda x: int(x.split(".")[0]))
            self.LRHS = os.listdir(os.path.join(self.root, "train", "LRHS"))
            self.LRHS.sort(key=lambda x: int(x.split(".")[0]))
            self.PAN = os.listdir(os.path.join(self.root, "train", "PAN"))
            self.PAN.sort(key=lambda x: int(x.split(".")[0]))

        if self.mode == "test":
            self.gtHS = os.listdir(os.path.join(self.root, 'test', "gtHS"))
            self.gtHS.sort(key=lambda x: int(x.split(".")[0]))
            self.LRHS = os.listdir(os.path.join(self.root, 'test', "LRHS"))
            self.LRHS.sort(key=lambda x: int(x.split(".")[0]))
            self.PAN = os.listdir(os.path.join(self.root, 'test', "PAN"))
            self.PAN.sort(key=lambda x: int(x.split(".")[0]))

    def __len__(self):
        return len(self.gtHS)

    def __getitem__(self, index):

        gt_hs, lr_hs, pan = self.gtHS[index], self.LRHS[index], self.PAN[index]
        data_ref = loadmat(os.path.join(self.root, self.mode, "gtHS", gt_hs))['da'].reshape(144, 160, 160)
        data_lrHS = loadmat(os.path.join(self.root, self.mode, "LRHS", lr_hs))['da'].reshape(144, 40, 40)
        data_Pan = loadmat(os.path.join(self.root, self.mode, "PAN", pan))['da'].reshape(1, 160, 160)
        return data_Pan, data_lrHS, data_ref


# def main():
#     root = 'D:\Pavia'
#     hs_train = My_dataset(root, 'train')
#     loader_train = DataLoader(hs_train, batch_size=8, shuffle=True, num_workers=8)
#     # for data_pan, data_LrHS, data_ref in loader_train:
#     #     print(data_pan.shape, data_LrHS.shape, data_ref.shape)
#
#
#
# if __name__ == '__main__':
#     main()
