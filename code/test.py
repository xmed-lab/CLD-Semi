import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='fully')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('-ep', '--epoch', type=int, required=True)
parser.add_argument('--speed', type=int, default=1)
parser.add_argument('-g', '--gpu', type=str,  default='0')
parser.add_argument('--cps', type=str, default=None)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from models.vnet import VNet
from utils import test_all_case, read_list, maybe_mkdir
from utils import config


if __name__ == '__main__':
    num_classes = 5
    stride_dict = {
        0: (32, 8),   # 1.5h
        1: (64, 16),  # 20min
        2: (128, 32), # 10min
    }
    stride = stride_dict[args.speed]

    snapshot_path = f'./logs/{args.exp}/'
    test_save_path = f'./logs/{args.exp}/predictions/ep_{args.epoch}'
    maybe_mkdir(test_save_path)

    model = VNet(
        n_channels=1, 
        n_classes=num_classes, 
        normalization='batchnorm', 
        has_dropout=False
    ).cuda()

    ckpt_path = os.path.join(snapshot_path, f'ckpts/ep_{args.epoch}.pth')
    if args.cps:
        model.load_state_dict(torch.load(ckpt_path)[args.cps])
    else: # for full-supervision
        model.load_state_dict(torch.load(ckpt_path))
    print(f'load checkpoint from {ckpt_path}')
    
    model.eval()
    with torch.no_grad():
        test_all_case(
            model, 
            read_list(args.split), 
            num_classes=num_classes,
            patch_size=config.patch_size, 
            stride_xy=stride[0], 
            stride_z=stride[1],
            test_save_path=test_save_path
        )
