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
    calculate_mse, calculate_test_metrics
)
import numpy as np
import matplotlib.pyplot as plt


def _visualize(hr: np.ndarray, sr_fixed: np.ndarray, sr_liif: np.ndarray, path: str):
    if hr.shape[0] == 2:  # 2 or 3 channels
        fig, ax = plt.subplots(3, 2, figsize=(8, 12))
        ax[0, 0].imshow(hr[0], cmap='jet')
        ax[0, 1].imshow(hr[1], cmap='jet')

        ax[1, 0].imshow(sr_fixed[0], cmap='jet')
        ax[1, 1].imshow(sr_fixed[1], cmap='jet')

        ax[2, 0].imshow(sr_liif[0], cmap='jet')
        ax[2, 1].imshow(sr_liif[1], cmap='jet')

        ax[0, 0].set_title('HR_U')
        ax[0, 1].set_title('HR_V')
        ax[1, 0].set_title('SR (fixed)_U')
        ax[1, 1].set_title('SR (fixed)_V')
        ax[2, 0].set_title('SR (arb)_U')
        ax[2, 1].set_title('SR (arb)_V')
    else:
        fig, ax = plt.subplots(3, 3, figsize=(12, 12))
        ax[0, 0].imshow(hr[0], cmap='jet')
        ax[0, 1].imshow(hr[1], cmap='jet')
        ax[0, 2].imshow(hr[2], cmap='jet')

        ax[1, 0].imshow(sr_fixed[0], cmap='jet')
        ax[1, 1].imshow(sr_fixed[1], cmap='jet')
        ax[1, 2].imshow(sr_fixed[2], cmap='jet')

        ax[2, 0].imshow(sr_liif[0], cmap='jet')
        ax[2, 1].imshow(sr_liif[1], cmap='jet')
        ax[2, 2].imshow(sr_liif[2], cmap='jet')

        ax[0, 0].set_title('HR_U')
        ax[0, 1].set_title('HR_V')
        ax[0, 2].set_title('HR_W')

        ax[1, 0].set_title('SR (fixed)_U')
        ax[1, 1].set_title('SR (fixed)_V')
        ax[1, 2].set_title('SR (fixed)_W')

        ax[2, 0].set_title('SR (arb)_U')
        ax[2, 1].set_title('SR (arb)_V')
        ax[2, 2].set_title('SR (arb)_W')

    plt.savefig(path)
    plt.close()


def visualize_outputs(
        i: int, hr: torch.Tensor, sr_fixed: torch.Tensor, sr_liif: torch.Tensor,
        mask_fixed: torch.Tensor, mask_liif: torch.Tensor, path_base: str
) -> int:
    hr = hr * mask_fixed
    sr_fixed = sr_fixed * mask_fixed
    sr_liif = sr_liif * mask_liif

    hr = hr.cpu().numpy()
    sr_fixed = sr_fixed.cpu().numpy()
    sr_liif = sr_liif.permute(0, 2, 1).reshape(*hr.shape).cpu().numpy()

    for j in range(hr.shape[0]):
        _visualize(hr[j], sr_fixed[j], sr_liif[j], f'{path_base}/{i}.png')
        i += 1
    return i


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def test_metrics(
        loader, model, min_v=None, max_v=None,
):
    model.eval()
    sr_fixed, hr_fixed, sr_liif, hr_liif, mask_fixed, mask_liif = [], [], [], [], [], []
    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        with torch.no_grad():
            if hasattr(model, 'forward_fixed'):
                model.encoder.return_upsample = True
                pred_fixed = model.forward_fixed(inp)
                model.encoder.return_upsample = False
            else:
                pred_fixed = torch.zeros_like(batch['gt_fixed'])
            pred_liif = model(inp, batch['coord'], batch['cell'])

        gt_fixed = batch['gt_fixed'].cpu().numpy()
        gt_liif = batch['gt'].cpu().numpy()

        gt_mask_fixed = batch['mask'].cpu().numpy()
        gt_mask_liif = batch['mask_liif'].cpu().numpy()

        sr_fixed.append(pred_fixed.cpu().numpy())
        sr_liif.append(pred_liif.cpu().numpy())
        hr_fixed.append(gt_fixed)
        hr_liif.append(gt_liif)
        mask_fixed.append(gt_mask_fixed)
        mask_liif.append(gt_mask_liif)
    
    sr_fixed = np.concatenate(sr_fixed, axis=0)
    sr_liif = np.concatenate(sr_liif, axis=0)
    hr_fixed = np.concatenate(hr_fixed, axis=0)
    hr_liif = np.concatenate(hr_liif, axis=0)
    mask_fixed = np.concatenate(mask_fixed, axis=0)
    mask_liif = np.concatenate(mask_liif, axis=0)
        
    sr_fixed = denormalize(sr_fixed, min_v, max_v)
    sr_liif = denormalize(sr_liif, min_v, max_v)
    hr_fixed = denormalize(hr_fixed, min_v, max_v)
    hr_liif = denormalize(hr_liif, min_v, max_v)
    
    res = calculate_test_metrics(sr_fixed, sr_liif, hr_fixed, hr_liif, mask_fixed, mask_liif)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.path + '/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('Config loaded')

    min_val = config['min_val']
    max_val = config['max_val']

    test_lr = np.load('./load/nm_wf/test_lr.npy')[:, :, 2:, :]
    test_hr = np.load('./load/nm_wf/test_hr.npy')[:, :, 12:, :]
    test_lr, test_hr, mask = preprocess_nan(test_lr, test_hr, 'test')
    test_lr = normalize(test_lr, min_val, max_val)
    test_hr = normalize(test_hr, min_val, max_val)

    test_inputs = {
        'lr_array': test_lr, 'hr_array': test_hr, 'mask_array': mask,
    }
    dataset = datasets.image_folder.PairedNMFolders(**test_inputs)
    dataset = datasets.wrappers.SRImplicitPairedTest(dataset)
    del test_inputs
    loader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)
    
    print('Loading model')
    pth_path = os.path.join(args.path, 'epoch-last.pth')
    torch_save = torch.load(pth_path)
    model_spec = torch_save['model']
    model = models.make(model_spec).cuda()
    model.load_state_dict(torch_save['model']['sd'])
    epoch = torch_save['epoch']

    print('Evaluating model')
    res = test_metrics(
        loader, model, min_v=min_val, max_v=max_val, 
    )
    config['test_result'] = res
    with open(os.path.join(args.path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    