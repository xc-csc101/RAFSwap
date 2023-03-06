import random
import numpy as np
from PIL import Image
import os

import torch
from torch.utils.data import Dataset
from utils import data_utils
import torch.nn.functional as F

class SwapImagesTrainDataset(Dataset):
    def __init__(self, root_img, root_mask, transform=None):
        super(SwapImagesTrainDataset, self).__init__()
        self.root_img = root_img
        self.root_mask = root_mask

        self.files_img = sorted(data_utils.make_dataset(self.root_img))
        self.files_mask = sorted(data_utils.make_dataset(self.root_mask))

        self.transform_img = transform

    def get_img(self, img_path):
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.transform_img(img)
        return img

    def get_mask(self, mask_path):
        mask = np.load(mask_path)
        mask = torch.from_numpy(mask)
        return mask

    def __getitem__(self, index):
        l = len(self.files_img)
        s_idx = index % l
        if index >= 4 * l:
            t_idx = s_idx
        else:
            t_idx = random.randrange(l)

        if t_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)
        # get image
        t_img = self.get_img(self.files_img[t_idx])
        s_img = self.get_img(self.files_img[s_idx])

        # get filenames
        t_img_n = self.files_img[t_idx].split('/')[-1].split('.')[0]
        s_img_n = self.files_img[s_idx].split('/')[-1].split('.')[0]

        # get mask
        t_mask = self.get_mask(self.files_mask[t_idx])
        s_mask = self.get_mask(self.files_mask[s_idx])

        return [t_img, t_img_n, t_mask], [s_img, s_img_n, s_mask], same

    def __len__(self):
        return len(self.files_img) * 5

class SwapImagesValDataset(Dataset):
    def __init__(self, root_img, root_mask, transform=None):
        super(SwapImagesValDataset, self).__init__()
        self.root_img = root_img
        self.root_mask = root_mask

        self.files_img = sorted(data_utils.make_dataset(self.root_img))
        self.files_mask = sorted(data_utils.make_dataset(self.root_mask))

        self.transform_img = transform

    def get_img(self, img_path):
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.transform_img(img)
        return img

    def get_mask(self, mask_path):
        mask = np.load(mask_path)
        mask = torch.from_numpy(mask)
        return mask

    def __getitem__(self, index):
        l = len(self.files_img)

        t_idx = index // l
        s_idx = index % l

        if t_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        # get image
        t_img = self.get_img(self.files_img[t_idx])
        s_img = self.get_img(self.files_img[s_idx])

        # get filenames
        t_img_n = self.files_img[t_idx].split('/')[-1].split('.')[0]
        s_img_n = self.files_img[s_idx].split('/')[-1].split('.')[0]

        # get mask
        t_mask = self.get_mask(self.files_mask[t_idx])
        s_mask = self.get_mask(self.files_mask[s_idx])

        return [t_img, t_img_n, t_mask], [s_img, s_img_n, s_mask], same

    def __len__(self):
        return len(self.files_img) * len(self.files_img)

class SwapImagesTxtDataset(Dataset):
    def __init__(self, root_img, root_mask, root_txt, transform=None, suffix='.jpg'):
        super(SwapImagesTxtDataset, self).__init__()
        self.root_img = root_img
        self.root_mask = root_mask
        self.root_txt = root_txt
        
        f = open(root_txt)
        file_pair = [s.strip() for s in f.readlines()]
        self.file_img_trg = [root_img + s.split('_')[0] + suffix for s in file_pair]
        self.file_img_src = [root_img + s.split('_')[1] + suffix for s in file_pair]

        self.file_mask_trg = [root_mask + s.split('_')[0] + '.npy' for s in file_pair]
        self.file_mask_src = [root_mask + s.split('_')[1] + '.npy' for s in file_pair]

        self.transform_img = transform

    def get_img(self, img_path):
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.transform_img(img)
        return img

    def get_mask(self, mask_path):
        mask = np.load(mask_path)
        mask = torch.from_numpy(mask)[:4,]
        return mask

    def __getitem__(self, index):
        # get image
        t_img = self.get_img(self.file_img_trg[index])
        s_img = self.get_img(self.file_img_src[index])

        # get filenames
        t_img_n = self.file_img_trg[index].split('/')[-1].split('.')[0]
        s_img_n = self.file_img_src[index].split('/')[-1].split('.')[0]

        # get mask
        t_mask = self.get_mask(self.file_mask_trg[index])
        s_mask = self.get_mask(self.file_mask_src[index])

        return [t_img, t_img_n, t_mask], [s_img, s_img_n, s_mask]

    def __len__(self):
        return len(self.file_img_trg)
