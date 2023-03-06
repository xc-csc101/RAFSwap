import os
from argparse import Namespace
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.faceswap_dataset import SwapImagesTxtDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
from kornia.morphology import dilation

def run():
    test_opts = TestOptions().parse()
    mode = 'vis'
    suffix = '.jpg'

    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_coupled = os.path.join(test_opts.exp_dir, mode, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_coupled = os.path.join(test_opts.exp_dir, mode, 'inference_coupled')

    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    root_img = opts.data_path
    root_mask = opts.mask_path
    root_txt = opts.txt_path
    dataset = SwapImagesTxtDataset(root_img = root_img,
                                   root_mask = root_mask,
                                   root_txt = root_txt,
                                   transform= transform
                                  )
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    for input_batch in tqdm(dataloader):
        [input_batch_t, t_name, t_mask], [input_batch_s, s_name, s_mask] = input_batch
        if global_i >= opts.n_images:
            break

        with torch.no_grad():
            input_cuda_t = input_batch_t.cuda().float()
            input_cuda_s = input_batch_s.cuda().float()
            xy = torch.cat([input_cuda_t, input_cuda_s], dim=0)
            t_mask, s_mask = t_mask.cuda().float(), s_mask.cuda().float()
            result_batch, m = run_on_batch(xy, t_mask, s_mask, net, opts)

        for i in range(opts.test_batch_size):
            mi = m[i]
            k = torch.ones(3, 3).cuda()
            mi = dilation(mi.unsqueeze(0), k).squeeze(0)

            result = tensor2im(result_batch[i] * mi + input_cuda_t[i] * (1 - mi))

            im_path_t = os.path.join(root_img, t_name[i] + suffix)
            im_path_s = os.path.join(root_img, s_name[i] + suffix)
            im_path = f'{t_name[i]}_{s_name[i]}' + suffix

            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch_t[i], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                target = Image.open(im_path_t)
                source = Image.open(im_path_s)
                if opts.resize_factors is not None:

                    res = np.concatenate([np.array(source.resize(resize_amount)),
                                          np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
                                          np.array(result.resize(resize_amount))], axis=1)
                else:
                    
                    res = np.concatenate([np.array(target.resize(resize_amount)),
                                          np.array(source.resize(resize_amount)),
                                          np.array(result.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))
            global_i += 1


def run_on_batch(inputs_xy, mask_t, mask_s, net, opts):
    result_batch, m = net.forward(inputs_xy, mask_t, mask_s, randomize_noise=False, resize=opts.resize_outputs)
    return result_batch, m


if __name__ == '__main__':
    ## run python inference_celeba.py
    run()
    print('down')
