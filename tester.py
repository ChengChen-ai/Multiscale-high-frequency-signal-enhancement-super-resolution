import os
import torch
import torchvision.utils as vutils
from readData.pairs_dataset_test import UnalignedDataLoader_test
from model.MMAFNet import MMAFNet
from utils import ops
import utils
import numpy as np
import cv2
from utils.config import Config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
use_gpu = torch.cuda.is_available()
rgb2Y = ops.RGB2YCbCr('rgb3ycbcr')
opt = Config('utils/training.yml')

data_loader = UnalignedDataLoader_test(opt)
val_loader = data_loader.load_data()

def HR_test():
    test_dir = os.path.join('./results', 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    model_restoration = MMAFNet()
    model_restoration.cuda()
    model_dir = './pretrained_model/Super_Resolution/models/MMAFNet'
    path_chk_rest1 = utils.get_last_path(model_dir, '_best.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest1)
    model_restoration.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_list = np.zeros(len(data_loader))
    i = 0

    psnr_val_rgb = []
    for batch_idx, data_val in enumerate((val_loader), 0):
        target = data_val['HR_images'].cuda()
        input_ = data_val['X4_images'].cuda()

        with torch.no_grad():
            start.record()
            restored = model_restoration(input_)
            end.record()
            torch.cuda.synchronize()
            time_list[i] = start.elapsed_time(end)  # milliseconds

        restored_HR = restored
        out_img = utils.tensor2np(restored.detach()[0])
        for res, tar in zip(restored_HR, target):
            psnr_val_rgb.append(utils.torchPSNR(res, tar))

        name = str(data_val['X4_paths']).split('/')[-1].split('x')[0] + 'x4.png'
        imAB_gen_file = os.path.join(test_dir, '{}'.format(name))
        # vutils.save_image(restored_HR.data, imAB_gen_file, normalize=True)
        cv2.imwrite(imAB_gen_file, out_img[:, :, [2, 1, 0]])
        i += 1
        print('processed item with idx: {}'.format(batch_idx))
        torch.cuda.empty_cache()

    psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
    print("-----------------------------------------------")
    print("PSNR:{}， TIME: {}ms".format(psnr_val_rgb, np.mean(time_list)))
    print("-----------------------------------------------")

if __name__ == '__main__':
    HR_test()

