import numpy as np
import torch

from tqdm.contrib.concurrent import process_map

from scipy.ndimage import zoom
from scipy import ndimage as nd

# from typing import Literal, Optional

import os


lat_obs = 34.5431
lon_obs = 125.8028
lat_grid = np.load('./load/nm_wf/lat_grid.npy')
lon_grid = np.load('./load/nm_wf/lon_grid.npy')


def get_obs_location(scale: float, org_scale: int = 6):
    rescale = scale / float(org_scale)
    rescaled_lat_grid = zoom(lat_grid, rescale, order=0)
    rescaled_lon_grid = zoom(lon_grid, rescale, order=0)

    lat_obs_idx = np.argmin(np.abs(rescaled_lat_grid - lat_obs))  # h
    lon_obs_idx = np.argmin(np.abs(rescaled_lon_grid - lon_obs))  # w

    return lat_obs_idx, lon_obs_idx


def fill_nan(arr: np.ndarray):
    # H W C(u,v,water_level)
    invalid_u = np.isnan(arr[:, :, 0])
    invalid_v = np.isnan(arr[:, :, 1])
    assert np.all(np.isclose(invalid_u, invalid_v))
    invalid = invalid_u

    idx = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)

    arr_result = np.zeros_like(arr)
    arr_result[:, :, 0] = arr[:, :, 0][tuple(idx)]  # U
    arr_result[:, :, 1] = arr[:, :, 1][tuple(idx)]  # V
    arr_result[:, :, 2] = arr[:, :, 2]              # Water Level
    return arr_result


def preprocess_nan(lr: np.ndarray, hr: np.ndarray, tag):
    """
    If just fill nan values using specific constant value(for example, 0),
    Convolutions will be affected by the constant value.
    So we fill nan values using nearest valid value.
    """
    color = 'blue' if tag == 'train' else 'green' if tag == 'val' else 'red'
    lr = np.where(lr < -1e+5, np.nan, lr)
    lr = np.stack(
        process_map(
            fill_nan, lr, max_workers=int(os.cpu_count() // 2), chunksize=1, desc=f"Filling NaN of {tag} LR",
            colour=color
        ),
        axis=0
    )
    mask = np.where(hr[:, :, :, :1] < -1e+5, 0., 1.).astype(float)
    hr = np.where(hr < -1e+5, 0., hr)
    return lr, hr, mask


def normalize(arr: np.ndarray, min_val: float, max_val: float):
    return (arr - min_val) / (max_val - min_val)


def denormalize(arr: np.ndarray, min_val: float, max_val: float):
    return arr * (max_val - min_val) + min_val


def preprocess_normalize(*arrs: np.ndarray, min_val: float, max_val: float):
    res = []
    for arr in arrs:
        res.append(normalize(arr, min_val, max_val))
    return res


def calculate_rmse(obs_u: np.ndarray, obs_v: np.ndarray, pred_u: np.ndarray, pred_v: np.ndarray):
    """
    Don't use this anymore.
    """
    rmse_u = np.sqrt(np.mean((obs_u - pred_u) ** 2))
    rmse_v = np.sqrt(np.mean((obs_v - pred_v) ** 2))
    obs_cspd = np.sqrt(obs_u ** 2 + obs_v ** 2)
    pred_cspd = np.sqrt(pred_u ** 2 + pred_v ** 2)
    rmse_cspd = np.sqrt(np.mean((obs_cspd - pred_cspd) ** 2))

    return rmse_u, rmse_v, rmse_cspd


def calculate_complex_corr(obs_u: np.ndarray, obs_v: np.ndarray, pred_u: np.ndarray, pred_v: np.ndarray):
    """
    Don't use this anymore.
    """
    x = (obs_u.T + 1j * obs_v.T).T
    y = (pred_u.T + 1j * pred_v.T).T

    x = x - np.mean(x)
    y = y - np.mean(y)
    norm_x = np.linalg.norm(x, ord=2)
    norm_y = np.linalg.norm(y, ord=2)

    abs_rho = np.abs(np.sum(x * np.conj(y)) / (norm_x * norm_y))
    return abs_rho


def calculate_mse(
        sr_fixed: torch.Tensor, sr_variable: torch.Tensor, hr_fixed: torch.Tensor, hr_variable: torch.Tensor,
        mask_fixed: torch.Tensor, mask_variable: torch.Tensor
) -> dict:
    sr_fixed = sr_fixed.cpu().numpy()
    sr_variable = sr_variable.cpu().numpy()
    hr_fixed = hr_fixed.cpu().numpy()
    hr_variable = hr_variable.cpu().numpy()
    mask_fixed = mask_fixed.cpu().numpy()
    mask_variable = mask_variable.cpu().numpy()

    div_fixed = np.sum(mask_fixed, axis=(1, 2, 3))
    div_variable = np.sum(mask_variable, axis=(1, 2))

    # L2 on fixed grid(U, V)
    l2_fixed_uv = np.mean(
        np.sum(np.square(sr_fixed[:, :2, :, :] - hr_fixed[:, :2, :, :]) * mask_fixed, axis=(1, 2, 3)) / div_fixed
    )

    # L2 on fixed grid(W)
    if sr_fixed.shape[1] == 3:
        l2_fixed_w = np.mean(
            np.sum(np.square(sr_fixed[:, 2:, :, :] - hr_fixed[:, 2:, :, :]) * mask_fixed, axis=(1, 2, 3)) / div_fixed
        )
    else:
        l2_fixed_w = 1e+10

    # L2 on variable grid(U, V)
    l2_variable_uv = np.mean(
        np.sum(np.square(sr_variable[:, :, :2] - hr_variable[:, :, :2]) * mask_variable, axis=(1, 2)) / div_variable
    )

    # L2 on variable grid(W)
    if sr_variable.shape[-1] == 3:
        l2_variable_w = np.mean(
            np.sum(np.square(sr_variable[:, :, 2:] - hr_variable[:, :, 2:]) * mask_variable, axis=(1, 2)) / div_variable
        )
    else:
        l2_variable_w = 1e+10

    res = {
        'l2_fixed_uv': float(l2_fixed_uv),
        'l2_fixed_w': float(l2_fixed_w),
        'l2_variable_uv': float(l2_variable_uv),
        'l2_variable_w': float(l2_variable_w)
    }
    return res


def calculate_test_metrics(
    sr_fixed: np.ndarray, sr_arb: np.ndarray, hr_fixed: np.ndarray, hr_arb: np.ndarray,
    mask_fixed: np.ndarray, mask_arb: np.ndarray
) -> dict:
    sr_fixed = np.where(mask_fixed < 0.5, np.nan, sr_fixed)
    sr_arb = np.where(mask_arb < 0.5, np.nan, sr_arb)
    hr_fixed = np.where(mask_fixed < 0.5, np.nan, hr_fixed)
    hr_arb = np.where(mask_arb < 0.5, np.nan, hr_arb)
    
    sr_fixed_uv = sr_fixed[:, :2, :, :].flatten()
    sr_fixed_uv = sr_fixed_uv[~np.isnan(sr_fixed_uv)]
    sr_fixed_w = sr_fixed[:, 2:, :, :].flatten()
    sr_fixed_w = sr_fixed_w[~np.isnan(sr_fixed_w)]
    sr_arb_uv = sr_arb[:, :, :2].flatten()
    sr_arb_uv = sr_arb_uv[~np.isnan(sr_arb_uv)]
    sr_arb_w = sr_arb[:, :, 2:].flatten()
    sr_arb_w = sr_arb_w[~np.isnan(sr_arb_w)]

    hr_fixed_uv = hr_fixed[:, :2, :, :].flatten()
    hr_fixed_uv = hr_fixed_uv[~np.isnan(hr_fixed_uv)]
    hr_fixed_w = hr_fixed[:, 2:, :, :].flatten()
    hr_fixed_w = hr_fixed_w[~np.isnan(hr_fixed_w)]
    hr_arb_uv = hr_arb[:, :, :2].flatten()
    hr_arb_uv = hr_arb_uv[~np.isnan(hr_arb_uv)]
    hr_arb_w = hr_arb[:, :, 2:].flatten()
    hr_arb_w = hr_arb_w[~np.isnan(hr_arb_w)]
    
    mse_fixed_uv = np.mean((sr_fixed_uv - hr_fixed_uv) ** 2)
    mse_fixed_w = np.mean((sr_fixed_w - hr_fixed_w) ** 2)
    mse_arb_uv = np.mean((sr_arb_uv - hr_arb_uv) ** 2)
    mse_arb_w = np.mean((sr_arb_w - hr_arb_w) ** 2)
    
    mae_fixed_uv = np.mean(np.abs(sr_fixed_uv - hr_fixed_uv))
    mae_fixed_w = np.mean(np.abs(sr_fixed_w - hr_fixed_w))
    mae_arb_uv = np.mean(np.abs(sr_arb_uv - hr_arb_uv))
    mae_arb_w = np.mean(np.abs(sr_arb_w - hr_arb_w))
    
    r2_fixed_uv = 1. - np.mean((sr_fixed_uv - hr_fixed_uv) ** 2) / np.mean((hr_fixed_uv - np.mean(hr_fixed_uv)) ** 2)
    r2_fixed_w = 1. - np.mean((sr_fixed_w - hr_fixed_w) ** 2) / np.mean((hr_fixed_w - np.mean(hr_fixed_w)) ** 2)
    r2_arb_uv = 1. - np.mean((sr_arb_uv - hr_arb_uv) ** 2) / np.mean((hr_arb_uv - np.mean(hr_arb_uv)) ** 2)
    r2_arb_w = 1. - np.mean((sr_arb_w - hr_arb_w) ** 2) / np.mean((hr_arb_w - np.mean(hr_arb_w)) ** 2)
    
    corr_fixed_uv = np.corrcoef(sr_fixed_uv, hr_fixed_uv.T)[0, 1]
    corr_fixed_w = np.corrcoef(sr_fixed_w, hr_fixed_w.T)[0, 1]
    corr_arb_uv = np.corrcoef(sr_arb_uv, hr_arb_uv.T)[0, 1]
    corr_arb_w = np.corrcoef(sr_arb_w, hr_arb_w.T)[0, 1]
    
    res = {
        'mse': {
            'fixed_uv': float(mse_fixed_uv),
            'fixed_w': float(mse_fixed_w),
            'arb_uv': float(mse_arb_uv),
            'arb_w': float(mse_arb_w)
        },
        'mae': {
            'fixed_uv': float(mae_fixed_uv),
            'fixed_w': float(mae_fixed_w),
            'arb_uv': float(mae_arb_uv),
            'arb_w': float(mae_arb_w)
        },
        'r2': {
            'fixed_uv': float(r2_fixed_uv),
            'fixed_w': float(r2_fixed_w),
            'arb_uv': float(r2_arb_uv),
            'arb_w': float(r2_arb_w)
        },
        'corr': {
            'fixed_uv': float(corr_fixed_uv),
            'fixed_w': float(corr_fixed_w),
            'arb_uv': float(corr_arb_uv),
            'arb_w': float(corr_arb_w)
        }
    }
    return res
