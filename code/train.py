import os
import sys
import logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='fully')
parser.add_argument('-s', '--split', type=str, default='noisy-train')
parser.add_argument('--split_eval', type=str, default='eval-50')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True) # <--
parser.add_argument('-ep', '--max_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('-g', '--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from models.vnet import VNet
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker
from utils.loss import DC_and_CE_loss, SoftDiceLoss
from data.transforms import RandomCrop, CenterCrop, RandomRotFlip, ToTensor
from data.knee_mri import KneeMRI, KneeMRI_light
from utils import config


if __name__ == '__main__':
    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'), 
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f'patch size: {config.patch_size}')

    # model
    model = VNet(
        n_channels=1, 
        n_classes=5, 
        normalization='batchnorm', 
        has_dropout=True
    ).cuda()

    # dataloader
    db_train = KneeMRI_light(split=args.split,
                       transform=transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(config.patch_size),
                          ToTensor(),
                       ]))
    db_eval = KneeMRI(split=args.split_eval,
                      transform = transforms.Compose([
                          CenterCrop(config.patch_size),
                          ToTensor()
                      ]))
    
    train_loader = DataLoader(
        db_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        worker_init_fn=seed_worker
    )
    eval_loader = DataLoader(db_eval, pin_memory=True)
    logging.info(f'{len(train_loader)} itertations per epoch')

    # optimizer, scheduler
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.base_lr, 
        momentum=0.9,
        weight_decay=1e-4
    )
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=1, 
        gamma=np.power(0.001, 1 / args.max_epoch)
    )

    # loss function
    loss_func = DC_and_CE_loss()

    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    for epoch_num in range(args.max_epoch + 1):
        loss_list = []

        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            image, label = fetch_data(batch)
            if args.mixed_precision:
                with autocast():
                    output = model(image)
                    del image
                    loss = loss_func(output, label)

                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()
            else:
                raise NotImplementedError

            loss_list.append(loss.item())

        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('loss', np.mean(loss_list), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}')

        lr_scheduler.step()

        if epoch_num % 20 == 0:
            save_path = os.path.join(snapshot_path, f'ckpts/ep_{epoch_num}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f'save model to {save_path}')

            # ''' ===== evaluation
            dice_list = [[] for _ in range(4)]
            model.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in eval_loader:
                image, gt = fetch_data(batch)
                output = model(image)
                
                shp = output.shape
                gt = gt.long()
                y_onehot = torch.zeros(shp).cuda()
                y_onehot.scatter_(1, gt, 1)

                x_onehot = torch.zeros(shp).cuda()
                output = torch.argmax(output, dim=1, keepdim=True).long()
                x_onehot.scatter_(1, output, 1)

                dice = dice_func(x_onehot, y_onehot, is_training=False)
                dice = dice.data.cpu().numpy()
                for i, d in enumerate(dice):
                    dice_list[i].append(d)
            
            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            # '''
    
    writer.close()
