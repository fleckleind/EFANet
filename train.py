import os
import sys
import copy
import warnings
warnings.filterwarnings("ignore")

import timm
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from utils import *
from engine import *
from models.efanet import EFANet
from configs.efanet import config_setting
from datasets.dataset import NPY_datasets


def load_best(model, best_model_dict):
    full_dict = copy.deepcopy(best_model_dict)
    for k in list(full_dict.keys()):
        if "total_" in k:
            del full_dict[k]
    model.load_state_dict(full_dict)


def load_weights_pre_best(pretrained_dict, model):
    full_dict = copy.deepcopy(pretrained_dict)
    model_dict = model.state_dict()
    for k in list(full_dict.keys()):
        if k in model_dict:
            if full_dict[k].shape != model_dict[k].shape:
                print("delete:{};shape model:{}".format(k, model_dict[k].shape))
                del full_dict[k]
    msg = model.load_state_dict(full_dict, strict=False)
    print("Pretrained Weights Loaded from former best version")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size, shuffle=True,
                              pin_memory=True, num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=config.num_workers, drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'egeunet':
        model = EGEUNet(num_classes=model_cfg['num_classes'], 
                        input_channels=model_cfg['input_channels'], 
                        c_list=model_cfg['c_list'], 
                        bridge=model_cfg['bridge'], gt_ds=model_cfg['gt_ds'],)
    else: raise Exception('network in not right!')
    model = model.cuda()
    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss, start_epoch, min_epoch = 999, 1, 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}.\
        resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader, model, criterion, optimizer, scheduler,
            epoch, step, logger, config, writer)

        loss = val_one_epoch(
            val_loader, model, criterion, epoch, logger, config)

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        if epoch % config.val_interval == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'wps-'+config.datasets+'-ep'+str(epoch)+'.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            val_loader, model, criterion, logger, config,)
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )      


if __name__ == '__main__':
    config = config_setting
    main(config)
