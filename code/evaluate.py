import os
import numpy as np
import argparse
from medpy import metric
from tqdm import tqdm

from utils import read_list, read_nifti
from utils import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_dir', type=str, default=None)
    args = parser.parse_args()

    test_cls = [1, 2, 3, 4]
    values = np.zeros((len(test_cls), 2)) # dice and asd
    ids_list = read_list('test')
    for data_id in tqdm(ids_list):
        pred = read_nifti(os.path.join(args.pred_dir, f'{data_id}.nii.gz'))
        label = read_nifti(os.path.join(config.base_dir, 'labelsTs', f'{data_id}.nii.gz'))
        for i in test_cls:
            pred_i = (pred == i)
            label_i = (label == i)
            if pred_i.sum() > 0 and label_i.sum() > 0:
                dice = metric.binary.dc(pred == i, label == i)
                asd = metric.binary.asd(pred == i, label == i)
                values[i - 1] += np.array([dice, asd])

    values /= len(ids_list)
    print(values)
    print(np.mean(values, axis=0))
