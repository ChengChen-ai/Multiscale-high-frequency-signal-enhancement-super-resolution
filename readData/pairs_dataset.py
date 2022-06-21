import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from readData.image_folder import ImageFolder
import argparse
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import warnings
from utils.ops import RGB2YCbCr


class PairedData(object):
    def __init__(self, data_loader_A):
        self.data_loader_A = data_loader_A

    def __iter__(self):
        self.data_loader_A_iter = iter(self.data_loader_A)
        return self

    def __next__(self):
        imgHR, imgX4, pathHR, pathX4 = next(self.data_loader_A_iter)

        return {'HR_images': imgHR, 'HR_paths': pathHR,
                'X4_images': imgX4, 'X4_paths': pathX4}


class UnalignedDataLoader_train(object):
    def __init__(self, params):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        dataset_train = torch.utils.data.DataLoader(
            ImageFolder(opt=params, root=params.TRAINING.TRAIN_DIR + '/' + 'DIV2K_train_HR', transform=transform,),
            batch_size=params.OPTIM.BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False, pin_memory=False)

        self.dataset_train = dataset_train
        self.paired_data = PairedData(self.dataset_train)

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../datasets/trainData', type=str)
    parser.add_argument('--width', default=128, type=int)
    parser.add_argument('--height', default=128, type=int)
    parser.add_argument('--load_size', default=142, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--phase', default='train', type=str)
    params = parser.parse_args()
    ycbcr = RGB2YCbCr('A')

    warnings.filterwarnings("error", category=UserWarning)
    unalignedDataLoader = UnalignedDataLoader_train(params)
    dataset = unalignedDataLoader.load_data()
    i = 0
    for _, u in enumerate(dataset):
        i += 1
        img_A = torchvision.utils.make_grid(u['X4_images']).numpy()
        img_B = torchvision.utils.make_grid(u['X4_YCbCr']).numpy()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 7))
        axes[0].imshow(np.transpose(img_A, (1, 2, 0)))
        axes[1].imshow(np.transpose(img_B, (1, 2, 0)))
        plt.show()


