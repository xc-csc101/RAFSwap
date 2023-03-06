import glob
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import cv2

mask_paths = sorted(glob.glob('../data/CelebA-HQ/train/labels/*'))
img_paths = '../data/CelebA-HQ/train/images/'
save_path = '../data/CelebA-HQ/train/label_npy/'
os.makedirs(save_path, exist_ok=True)

for idx, mask_path in enumerate(mask_paths):
    img_name = os.path.basename(mask_path).replace('.png', '.jpg')
    img = cv2.imread(img_paths + img_name)

    mask = Image.open(mask_path)
    mask = np.array(mask)
    mask = torch.from_numpy(mask)
    mask = F.interpolate(
            mask.view(1, 1, 512, 512).float(),
            (256, 256),
            mode="nearest")
    mask = mask.type(torch.uint8)
    
    mask_lip = (mask == 11).float() + (mask == 12).float()   # mask up lip and down lip
    mask_nose = (mask == 2).float()  # mask nose
    mask_brow = (mask == 6).float() + (mask == 7).float()
    mask_eyes = (mask == 4).float() + (mask == 5).float()
    mask_list = [mask_lip, mask_nose, mask_brow, mask_eyes]

    mask_aug = torch.cat(mask_list, 0)  # (4, 1, h, w)
    mask_re = F.interpolate(mask_aug, size=64)
    np.save(save_path + img_name.replace('jpg', 'npy'), mask_re.numpy())
    print(f'{idx}/{len(mask_paths)}', end='\r')
