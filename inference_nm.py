import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from nm_test_funcs import (
    get_obs_location, denormalize, calculate_rmse, calculate_complex_corr, normalize, preprocess_nan,
)
import numpy as np
import matplotlib.pyplot as plt

import PIL


def _save_image(hr, bicubic_arr, liif_arr, name, path):
    hr_max = np.nanmax(hr)
    hr_min = np.nanmin(hr)
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].pcolormesh(hr, cmap='jet', vmin=hr_min, vmax=hr_max)
    ax[0].set_title(f'HR_{name} / shape: {hr.shape}')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].pcolormesh(bicubic_arr, cmap='jet', vmin=hr_min, vmax=hr_max)
    ax[1].set_title(f'Bicubic_{name} / shape: {bicubic_arr.shape}')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].pcolormesh(liif_arr, cmap='jet', vmin=hr_min, vmax=hr_max)
    ax[2].set_title(f'LIIF_{name} / shape: {liif_arr.shape}')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--scale', type=int)
    parser.add_argument('--inference_split', type=int, default=10)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    
    with open(args.path + '/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('Config loaded')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    i_s = args.inference_split
    
    img_save_path = args.path + f'/inference/scale_{args.scale}'
    os.makedirs(img_save_path, exist_ok=True)
    
    min_val = config['min_val']
    max_val = config['max_val']
    
    print('Loading data...')
    test_lr = np.load('./load/nm_wf/test_lr.npy')[:, :, 2:, :]
    test_hr = np.load('./load/nm_wf/test_hr.npy')[:, :, 12:, :]
    test_lr, test_hr, mask = preprocess_nan(test_lr, test_hr, 'test')
    test_lr = normalize(test_lr, min_val, max_val)
    test_inputs = {
        'lr_array': test_lr, 'hr_array': test_hr, 'mask_array': mask,
    }
    dataset = datasets.image_folder.PairedNMFolders(**test_inputs)
    dataset = datasets.wrappers.SRImplicitPairedTest(dataset=dataset, scale=args.scale)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    del test_inputs
    del test_hr
    del test_lr
    
    print('Loading model...')
    path_path = os.path.join(args.path, 'epoch-last.pth')
    torch_save = torch.load(path_path)
    model_spec = torch_save['model']
    model = models.make(model_spec).cpu()
    model.load_state_dict(torch_save['model']['sd'])
    model.eval()
    model.encoder.return_upsample = False
    model.cuda()

    print('Inference...')
    pbar = tqdm(loader, leave=False, desc='Inference')
    i = 0
    for batch in pbar:
        inp = batch['inp'].cuda()
        coord = batch['coord'].cuda()
        cell = batch['cell'].cuda()
        hr = batch['gt_fixed'].permute(0, 2, 3, 1).numpy()
        hr_mask = batch['mask'].permute(0, 2, 3, 1).numpy()

        with torch.no_grad():
            pred = []
            n_pixels = coord.shape[1]
            for n in range(i_s):
                pred.append(
                    model(
                        inp, coord[:, n * n_pixels // i_s:(n + 1) * n_pixels // i_s], cell[:, n * n_pixels // i_s:(n + 1) * n_pixels // i_s]
                    ).cpu()
                )
            pred = torch.cat(pred, dim=1)
        
        bicubic_pred = batch['sr_bicubic'].permute(0, 2, 3, 1).numpy()
        mask = batch['mask_up'].permute(0, 2, 3, 1).numpy()
        liif_pred = pred.reshape(*bicubic_pred.shape).numpy()
        
        bicubic_pred = denormalize(bicubic_pred, min_val, max_val)
        liif_pred = denormalize(liif_pred, min_val, max_val)
        
        hr = np.where(hr_mask < 0.5, np.nan, hr)
        bicubic_pred = np.where(mask < 0.5, np.nan, bicubic_pred)[0]
        liif_pred = np.where(mask < 0.5, np.nan, liif_pred)[0]
        
        os.makedirs(img_save_path + f'/{i}', exist_ok=True)
        _save_image(hr[0, :, :, 0], bicubic_pred[:, :, 0], liif_pred[:, :, 0], 'u', img_save_path + f'/{i}/u.png')
        _save_image(hr[0, :, :, 1], bicubic_pred[:, :, 1], liif_pred[:, :, 1], 'v', img_save_path + f'/{i}/v.png')
        _save_image(hr[0, :, :, 2], bicubic_pred[:, :, 2], liif_pred[:, :, 2], 'w', img_save_path + f'/{i}/w.png')
        
        np.save(img_save_path + f'/{i}/hr.npy', hr[0])
        np.save(img_save_path + f'/{i}/bicubic.npy', bicubic_pred)
        np.save(img_save_path + f'/{i}/liif.npy', liif_pred)

        i = i + 1
