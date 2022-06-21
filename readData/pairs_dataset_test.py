import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from readData.image_folder_test import ImageFolder


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


class UnalignedDataLoader_test(object):
    def __init__(self, params):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        dataset_train = torch.utils.data.DataLoader(
            ImageFolder(opt=params, root=params.TRAINING.VAL_DIR + '/' + 'DIV2K_valid_HR', transform=transform,),
            batch_size=1,
            num_workers=8, shuffle=False, drop_last=False, pin_memory=True)

        self.dataset_train = dataset_train
        self.paired_data = PairedData(self.dataset_train)

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset_train)