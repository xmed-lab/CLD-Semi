import os
import math
from tqdm import tqdm
import numpy as np
import random
import SimpleITK as sitk

import torch
import torch.nn.functional as F

from utils import config


def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    itk_arr = sitk.GetArrayFromImage(itk_img)
    return itk_arr


def read_list(split):
    ids_list = np.loadtxt(
        os.path.join(config.base_dir, 'splits', f'{split}.txt'), 
        dtype=str
    ).tolist()
    return sorted(ids_list)


def read_data(data_id, nifti=False, test=False, normalize=False):
    if not nifti: # load npy files
        im_path = os.path.join(config.base_dir, 'npy', f'{data_id}_image.npy')
        lb_path = os.path.join(config.base_dir, 'npy', f'{data_id}_label.npy')
        if not os.path.exists(im_path) or not os.path.exists(lb_path):
            raise ValueError(data_id)
        image = np.load(im_path)
        label = np.load(lb_path)
    else:
        tag = 'Tr' if not test else 'Ts'
        image = read_nifti(os.path.join(config.base_dir, f'images{tag}', f'{data_id}_0000.nii.gz'))
        label = read_nifti(os.path.join(config.base_dir, f'labels{tag}', f'{data_id}.nii.gz'))
    
    if normalize:
        image = (image - image.mean()) / (image.std() + 1e-8)
        image = image.astype(np.float32)
    
    return image, label


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fetch_data(batch, labeled=True):
    image = batch['image'].cuda()
    if labeled:
        label = batch['label'].cuda().unsqueeze(1)
        return image, label
    else:
        return image


def test_all_case(net, ids_list, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, test=True, normalize=True)
        pred, _ = test_single_case(
            net, 
            image, 
            stride_xy, 
            stride_z, 
            patch_size, 
            num_classes=num_classes
        )
        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes):
    image = image.transpose(2, 1, 0) # <-- take care the shape
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
                test_patch = test_patch.transpose(2, 4)
                y1 = net(test_patch) # <--
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    
    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map
