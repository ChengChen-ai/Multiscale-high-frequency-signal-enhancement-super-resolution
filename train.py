import os
import torch
import random
import time
import numpy as np
import utils
from readData.pairs_dataset import UnalignedDataLoader_train
from readData.pairs_dataset_vali import UnalignedDataLoader_val
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import losses
from warmup_scheduler import GradualWarmupScheduler
# from utils.visualisations import Visualizer
from utils.config import Config
from utils import util
from model.MMAFNet import MMAFNet
from collections import OrderedDict
opt = Config('utils/training.yml')
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def train():
    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    start_epoch = 1

    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    ######### Reveal ###########
    # visualizer = Visualizer(opt)

    ####load model
    model_restoration = MMAFNet()
    model_restoration.cuda()

    print("\n=================parameters======================")
    print(model_restoration.parameters())
    print(sum(param.numel() for param in model_restoration.parameters()))

    new_lr = opt.OPTIM.LR_INITIAL
    restoration_optim = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    ######### Scheduler ###########
    #warmup_epochs = 10
    #scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(restoration_optim, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
    #                                                        eta_min=opt.OPTIM.LR_MIN)
    #scheduler = GradualWarmupScheduler(restoration_optim, multiplier=1, total_epoch=warmup_epochs,
    #                                   after_scheduler=scheduler_cosine)
    ######### Resume ###########
    if opt.TRAINING.RESUME:
        path_chk_rest1 = utils.get_last_path(model_dir, '_best.pth')
        utils.load_checkpoint(model_restoration, path_chk_rest1)
        start_epoch = utils.load_start_epoch(path_chk_rest1) + 1
        utils.load_optim(restoration_optim, path_chk_rest1)

        #for i in range(1, start_epoch):
        #    scheduler.step()
        #new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    device_ids = [i for i in range(torch.cuda.device_count())]
    if len(device_ids) > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
        # model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()
    criterion_lsap = losses.L_spa()

    ######### DataLoaders ###########
    train_dataset = UnalignedDataLoader_train(opt)
    val_dataset = UnalignedDataLoader_val(opt)
    val_loader = val_dataset.load_data()

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        train_loader = train_dataset.load_data()
        epoch_loss = 0
        model_restoration.train()

        for i, data in enumerate(tqdm(train_loader), 0):
            targetHR = data['HR_images'].cuda()
            lrX4 = data['X4_images'].cuda()

            restored = model_restoration(lrX4)
            loss_char = criterion_char(restored, targetHR)
            loss_edge = criterion_edge(restored, targetHR)
            loss_lsap = torch.mean(criterion_lsap(targetHR, restored))

            RGB_loss = loss_char + (0.05 * loss_edge) + (0.5 * loss_lsap)

            restoration_optim.zero_grad()
            RGB_loss.backward()
            restoration_optim.step()

            epoch_loss += RGB_loss.item()

            #### Evaluation ####
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            psnr_val_rgb = 0
            ssim_val_rgb = 0
            psnr_val_rgb_every = []
            number = 0
            for ii, data_val in enumerate((val_loader), 0):
                target = data_val['HR_images'].cuda()
                input_ = data_val['X4_images'].cuda()

                with torch.no_grad():
                    restored = model_restoration(input_)

                sr_img = util.tensor2np(restored.detach()[0])
                gt_img = util.tensor2np(target.detach()[0])

                for res,tar in zip(restored,target):
                    psnr_val_rgb_every.append(utils.torchPSNR(res, tar))

                if epoch % 20 == 0:
                    psnr_val_rgb += util.compute_psnr(sr_img, gt_img)
                    ssim_val_rgb += util.compute_ssim(sr_img, gt_img)
                    number += 1

            psnr_val_rgb_every = torch.stack(psnr_val_rgb_every).mean().item()

            if psnr_val_rgb_every > best_psnr:
                best_psnr = psnr_val_rgb_every
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'MMAF_optimizer': restoration_optim.state_dict()
                            }, os.path.join(model_dir, "model_MMAF_best.pth"))

            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb_every, best_epoch, best_psnr))
            if epoch % 20 == 0:
                avg_psnr = psnr_val_rgb / number
                avg_ssim = ssim_val_rgb / number
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'MMAF_optimizer': restoration_optim.state_dict()
                            }, os.path.join(model_dir, "model_MMAF_epoch_"+ str(epoch) + ".pth"))

                print("[epoch %d PSNR: %.4f SSIM: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, avg_psnr, avg_ssim,
                                                                                             best_epoch, best_psnr))
        #scheduler.step()
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, new_lr))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'MMAF_optimizer': restoration_optim.state_dict()
                    }, os.path.join(model_dir, "model_MMAFlatest.pth"))


if __name__ == '__main__':
    train()
