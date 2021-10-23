import os

import torch
import yaml

from utils import network_parameters
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import utils
import numpy as np
import random
from data_RGB import get_training_data, get_validation_data

from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model

## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

## Load yaml configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']

## Build Model
print('==> Build the model')
model_restored = SUNet_model(opt)
p_number = network_parameters(model_restored)
model_restored.cuda()

## Training model path direction
mode = opt['MODEL']['MODE']

model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

## GPU
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

## Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

## Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

## Loss
L1_loss = nn.L1Loss()

## DataLoaders
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=0, drop_last=False)
val_dataset = get_validation_data(val_dir, {'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False)

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restored.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        # Forward propagation
        for param in model_restored.parameters():
            param.grad = None
        target = data[0].cuda()
        input_ = data[1].cuda()
        restored = model_restored(input_)

        # Compute loss
        #loss = Charbonnier_loss(restored, target)
        loss = L1_loss(restored, target)

        # Back propagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    ## Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            with torch.no_grad():
                restored = model_restored(input_)

            for res, tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))
                ssim_val_rgb.append(utils.torchSSIM(restored, target))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

        # Save the best PSNR model of validation
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

        # Save the best SSIM model of validation
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSSIM.pth"))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

        """ 
        # Save evey epochs of model
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        """

        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
        writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
