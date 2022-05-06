import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import read_list, read_data


class KneeMRI_light(Dataset):
    def __init__(self, split='train', repeat=None, transform=None, unlabeled=False):
        self.ids_list = read_list(split)
        self.repeat = repeat
        if self.repeat is None:
            self.repeat = len(self.ids_list)
        print('total {} datas'.format(self.repeat))
        self.transform = transform
        self.unlabeled = unlabeled
        self.num_cls = 5
        self._weight = None

    def __len__(self):
        return self.repeat

    def _get_data(self, data_id):
        # [160, 384, 384]
        image, label = read_data(data_id)
        return data_id, image, label

    @property
    def weight(self):
        if self.unlabeled:
            raise ValueError
        
        if self._weight is not None:
            return self._weight
        
        weight = np.zeros(self.num_cls)
        for data_id in self.ids_list:
            _, _, label = self._get_data(data_id)
            label = label.reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            weight += tmp

        weight = weight.astype(np.float32)
        weight = weight / np.sum(weight)
        self._weight = np.power(np.amax(weight) / weight, 1/3)
        return self._weight

    def __getitem__(self, index):
        index = index % len(self.ids_list)
        data_id = self.ids_list[index]
        _, image, label = self._get_data(data_id)
        if self.unlabeled: # <-- for safety
            label[:] = 0

        image = (image - image.mean()) / (image.std() + 1e-8)
        image = image.astype(np.float32)
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        return sample


class KneeMRI(KneeMRI_light):
    def __init__(self, split='train', repeat=None, transform=None, unlabeled=False):
        super().__init__(split=split, repeat=repeat, transform=transform, unlabeled=unlabeled)
        self.data_list = {}
        for data_id in tqdm(self.ids_list): # <-- load data to memory
            image, label = read_data(data_id)
            self.data_list[data_id] = (image, label)

    def _get_data(self, data_id):
        image, label = self.data_list[data_id]
        return data_id, image, label
