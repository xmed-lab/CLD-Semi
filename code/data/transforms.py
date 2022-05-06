import torch
import numpy as np


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        padding_flag = image.shape[0] <= self.output_size[0] or \
                       image.shape[1] <= self.output_size[1] or \
                       image.shape[2] <= self.output_size[2]

        # pad the sample if necessary
        if padding_flag:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)

        w1, h1, d1 = None, None, None
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if padding_flag:
                item = np.pad(item, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if w1 is None:
                (w, h, d) = item.shape
                w1 = int(round((w - self.output_size[0]) / 2.))
                h1 = int(round((h - self.output_size[1]) / 2.))
                d1 = int(round((d - self.output_size[2]) / 2.))
            item = item[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            ret_dict[key] = item
        
        return ret_dict


class RandomCrop(object):
    '''
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    '''
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        padding_flag = image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]

        # pad the sample if necessary
        if padding_flag:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
        
        w1, h1, d1 = None, None, None
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if padding_flag:
                item = np.pad(item, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if w1 is None:
                (w, h, d) = item.shape
                w1 = np.random.randint(0, w - self.output_size[0])
                h1 = np.random.randint(0, h - self.output_size[1])
                d1 = np.random.randint(0, d - self.output_size[2])
            item = item[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            ret_dict[key] = item
        
        return ret_dict


class RandomCrop_cz(object):
    def __init__(self, output_size, span, p=1):
        self.p = p
        self.span = span
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        padding_flag = image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]

        # pad the sample if necessary
        if padding_flag:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
        
        w1, h1, d1 = None, None, None
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if padding_flag:
                item = np.pad(item, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if w1 is None:
                (w, h, d) = item.shape
                if np.random.randint(self.p + 1) >= 1:
                    w1 = np.random.randint(self.span, w - self.output_size[0] - self.span)
                else:
                    w1 = np.random.randint(0, w - self.output_size[0])
                h1 = np.random.randint(0, h - self.output_size[1])
                d1 = np.random.randint(0, d - self.output_size[2])
            item = item[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            ret_dict[key] = item
        
        return ret_dict

class RandomRotFlip(object):
    '''
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    '''
    def __call__(self, sample):
        k = np.random.randint(0, 4)
        axis = np.random.randint(0, 2)
        
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            item = np.rot90(item, k)
            item = np.flip(item, axis=axis).copy()
            ret_dict[key] = item
        
        return ret_dict


class RandomNoise(object):
    def __init__(self,sigma=0.01):
        self.sigma = sigma

    def __call__(self, sample):
        ret_dict = {}
        for key in sample.keys():
            if key == 'image':
                image = sample[key]
                noise = np.clip(
                    self.sigma * np.random.randn(*image.shape),
                    -2 * self.sigma, 
                     2 * self.sigma
                )
                noise = noise + self.mu
                image = image + noise
                ret_dict[key] = image
            else:
                ret_dict[key] = sample[key]
        
        return ret_dict


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if key == 'image':
                ret_dict[key] = torch.from_numpy(item).unsqueeze(0).float()
            elif key == 'label':
                ret_dict[key] = torch.from_numpy(item).long()
            else:
                raise ValueError(key)
        
        return ret_dict
