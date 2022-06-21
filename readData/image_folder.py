import torch.utils.data as data
from sklearn.utils import shuffle
from PIL import Image
import os
import os.path
import torchvision.transforms.functional as TF
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path)

def make_dataset(dir):
    image_pathsX4 = []
    image_pathsHR = []
    for root, _, fnames in sorted(os.walk(dir)):
        fnames = shuffle(fnames, n_samples=len(fnames))
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                image_pathsHR.append(path)
                image_pathsX4.append(path.replace("DIV2K_train_HR", "DIV2K_train_LR_bicubic_X4").split('.png')[0] + 'x4.png')

    return image_pathsHR, image_pathsX4

class ImageFolder(data.Dataset):
    def __init__(self, opt, root, transform=None, return_paths=True,
                 loader=default_loader):

        image_pathsHR, image_pathsX4 = make_dataset(root)

        if (len(image_pathsHR) or len(image_pathsX4)) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.image_pathsHR = image_pathsHR
        self.image_pathsX4 = image_pathsX4
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

        self.ps = opt.TRAINING.TRAIN_PS

    def __getitem__(self, index):
        pathHR = self.image_pathsHR[index]
        pathX4 = self.image_pathsX4[index]
        imgHR = self.loader(pathHR)
        imgX4 = self.loader(pathX4)

        ps = self.ps
        X4_w, X4_h = imgX4.size

        padw = ps-X4_w if X4_w<ps else 0
        padh = ps-X4_h if X4_w<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            imgX4 = TF.pad(imgX4, (0,0,padw,padh), padding_mode='reflect')
            imgHR = TF.pad(imgHR, (0, 0, padw * 4, padh * 4), padding_mode='reflect')

        if self.transform is not None:
            imgHR = self.transform(imgHR)
            imgX4 = self.transform(imgX4)

        hh, ww = imgX4.shape[1], imgX4.shape[2]

        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)

        # Crop patch
        imgX4 = imgX4[:, rr:rr+ps, cc:cc+ps]
        imgHR = imgHR[:, rr*4:(rr + ps)*4, cc*4:(cc + ps)*4]

        if self.return_paths:
            return imgHR, imgX4, pathHR, pathX4
        else:
            return imgHR, imgX4, '', ''

    def __len__(self):
        return len(self.image_pathsX4)


if __name__ == '__main__':
    path = "../datasets/trainData/DIV2K_train_HR"
    imageFolder = ImageFolder(path)

