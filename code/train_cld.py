import os
import sys
import logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='cld')
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)
parser.add_argument('-ep', '--max_epoch', type=int, default=300)
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=0.1)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=True)
parser.add_argument('--crop_z', type=int, default=20)
parser.add_argument('--confs_iter', type=int, default=2)
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from models.vnet import VNet
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss, WeightedCrossEntropyLoss
from data.transforms import RandomCrop_cz, CenterCrop, RandomRotFlip, ToTensor
from data.knee_mri import KneeMRI, KneeMRI_light
from utils import config


def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epoch
        return args.cps_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.cps_w


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)


def make_loader(split, dst_cls=KneeMRI, repeat=None, is_training=True, unlabeled=False):
    if is_training:
        dst = dst_cls(
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            transform=transforms.Compose([
                RandomRotFlip(),
                RandomCrop_cz(config.patch_size, span=args.crop_z, p=1),
                ToTensor()
            ])
        )
        return DataLoader(
            dst, 
            batch_size=args.batch_size, 
            shuffle=True,  
            num_workers=args.num_workers, 
            pin_memory=True, 
            worker_init_fn=seed_worker
        )
    else:
        dst = dst_cls(
            split=split,
            transform=transforms.Compose([
                CenterCrop(config.patch_size),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)


def make_model_all():
    model = VNet(
        n_channels=1, 
        n_classes=5, 
        normalization='batchnorm', 
        has_dropout=True
    ).cuda()
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
    return model, optimizer, lr_scheduler


def cal_sampling_mask(output, sam_rate, min_sr=0.0):
    pred_map = torch.argmax(output, dim=1).float()
    sample_map = torch.zeros_like(pred_map).float()
    vol_shape = pred_map.shape
    for idx in range(5):
        prob = 1 - sam_rate[idx]
        if idx >= 1 and prob > (1 - min_sr):
            prob = (1 - min_sr)
        rand_map = torch.rand(vol_shape).cuda() * (pred_map == idx) 
        rand_map = (rand_map > prob) * 1.0
        sample_map += rand_map
    return sample_map


def cal_confidence(output, label):
    new_conf = torch.zeros(5).float().cuda()
    scores = F.softmax(output.detach(), dim=1)
    for ind in range(5):
        cat_mask_sup_gt = (label == ind).squeeze(1)
        conf_map_sup = scores[:, ind, ...]
        value = torch.sum(conf_map_sup * cat_mask_sup_gt) / (torch.sum(cat_mask_sup_gt) + 1e-12)
        new_conf[ind] = value
    return new_conf


class ConfidenceBank:
    def __init__(self, confs=None, num_cls=5, momentum=0.999, update_iter=1):
        self.num_cls = num_cls
        self.momentum = momentum
        self.update_iter = update_iter
        self.tot_iter = 0
        if confs is None:
            self.confs = torch.rand(self.num_cls).float().cuda()
        else:
            self.confs = torch.clone(confs)
        self.confs_all = torch.zeros(self.num_cls).float().cuda()
        self.confs_cnt = torch.zeros(self.num_cls).float().cuda()

    def record_data(self, output, label):
        scores = F.softmax(output.detach(), dim=1)
        for ind in range(5):
            cat_mask_sup_gt = (label == ind).squeeze(1) # <--
            conf_map_sup = scores[:, ind, ...]
            self.confs_all[ind] += torch.sum(conf_map_sup * cat_mask_sup_gt)
            self.confs_cnt[ind] += torch.sum(cat_mask_sup_gt).float()
        
        self.tot_iter += 1
        if self.tot_iter % self.update_iter == 0:
            new_confs = self.confs_all / (self.confs_cnt + 1e-12)
            if self.tot_iter <= self.update_iter: # first update
                self.confs = new_confs
            else:
                self.confs = self.confs * self.momentum + new_confs * (1 - self.momentum)

            self.confs_all = torch.zeros(self.num_cls).float().cuda()
            self.confs_cnt = torch.zeros(self.num_cls).float().cuda()

    def get_confs(self):
        return self.confs


def cal_sampling_rate(confs, gamma=0.5):
    sam_rate = (1 - confs)
    sam_rate = sam_rate / (torch.max(sam_rate) + 1e-12)
    sam_rate = sam_rate ** gamma
    return sam_rate


def cal_weights(pos_masks, label):
    b, _, w, h, d = label.shape
    weight = torch.zeros((b, w, h, d)).float().cuda()
    for i in range(5):
        weight += pos_masks[:, i, ...] * (label[:, 0, ...] == i)
    return weight


if __name__ == '__main__':
    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    
    # make logger
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

    # make data loader
    unlabeled_loader = make_loader(args.split_unlabeled, dst_cls=KneeMRI_light, unlabeled=True)
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset))
    eval_loader = make_loader(args.split_eval, is_training=False)

    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')
    
    # make model, optimizer, and lr scheduler
    model_A, optimizer_A, lr_scheduler_A = make_model_all()
    model_B, optimizer_B, lr_scheduler_B = make_model_all()
    model_A = kaiming_normal_init_weight(model_A)
    model_B = xavier_normal_init_weight(model_B)

    # make loss function
    weight = labeled_loader.dataset.weight
    loss_func = make_loss_function(args.sup_loss, weight=weight)
    if args.cps_loss == 'wce':
        cps_loss_func = WeightedCrossEntropyLoss(weight=weight)
    else:
        raise ValueError

    # confidence bank
    confs_A = ConfidenceBank(num_cls=5, momentum=0.999, update_iter=args.confs_iter)
    confs_B = ConfidenceBank(confs=confs_A.get_confs(), num_cls=5, momentum=0.999, update_iter=args.confs_iter) # <--

    if args.mixed_precision:
        amp_grad_scaler = GradScaler()
    
    cps_w = get_current_consistency_weight(0)
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_cps_list = []
        loss_sup_list = []

        model_A.train()
        model_B.train()
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()

            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image.shape[0] // 2

            if args.mixed_precision:
                with autocast():
                    output_A = model_A(image)
                    output_B = model_B(image)
                    del image
                    
                    # split labeled and unlabeled output
                    output_A_l = output_A[:tmp_bs, ...]
                    output_B_l = output_B[:tmp_bs, ...]
                    
                    # sup (ce + dice)
                    # <--
                    # update confidence bank
                    confs_A.record_data(output_A_l, label_l)
                    confs_B.record_data(output_B_l, label_l)
                    # <--
                    # loss function
                    loss_sup = loss_func(output_A_l, label_l) + loss_func(output_B_l, label_l)

                    # cps (ce only)
                    max_A = torch.argmax(output_A.detach(), dim=1, keepdim=True).long()
                    max_B = torch.argmax(output_B.detach(), dim=1, keepdim=True).long()
                    # <-- 
                    # update category sampling rate
                    sam_rate_A = cal_sampling_rate(confs_A.get_confs())
                    sam_rate_B = cal_sampling_rate(confs_B.get_confs())
                    # <--
                    # update sampling mask (without gt)
                    sample_map_A = cal_sampling_mask(output_A, sam_rate_A)
                    sample_map_B = cal_sampling_mask(output_B, sam_rate_B)
                    # <--
                    # loss function
                    loss_cps = cps_loss_func(output_A, max_B, sample_map_A) + cps_loss_func(output_B, max_A, sample_map_B)

                    # loss prop
                    loss = loss_sup + cps_w * loss_cps

                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer_A)
                amp_grad_scaler.step(optimizer_B)
                amp_grad_scaler.update()
            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(loss_cps.item())

        writer.add_scalar('lr', get_lr(optimizer_A), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}')
        logging.info(f'    - confidence {confs_A.get_confs()}, {confs_B.get_confs()}')
        logging.info(f'    - sampling rate {sam_rate_A}, {sam_rate_B}')

        lr_scheduler_A.step()
        lr_scheduler_B.step()
        cps_w = get_current_consistency_weight(epoch_num)

        if epoch_num % 20 == 0:
            save_path = os.path.join(snapshot_path, f'ckpts/ep_{epoch_num}.pth')
            torch.save({
                'A': model_A.state_dict(), 
                'B': model_B.state_dict()
            }, save_path)
            logging.info(f'save model to {save_path}')

            # ''' ===== evaluation
            dice_list = [[] for _ in range(4)]
            model_A.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    output = model_A(image)
                    del image
                
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
