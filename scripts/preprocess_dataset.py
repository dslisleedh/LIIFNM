import h5py
import numpy as np
import pandas as pd

from tqdm.contrib.concurrent import process_map
from datetime import datetime
import os
import natsort
from multiprocessing import cpu_count


def get_feats_from_file(file_name):
    f_h5 = h5py.File(file_name, 'r')
    sample_names = list(map(lambda x: x[4:], list(np.array(f_h5.get('/Time')))))
    if '_12_' in file_name:  # This might cause problem if file name is changed or duplicated problem is solved.
        sample_names = sample_names[1:]

    feats = []
    times = []
    for sample_name in sample_names:
        u = np.array(f_h5.get('Results/velocity U/velocity U' + sample_name))[0].T
        v = np.array(f_h5.get('Results/velocity V/velocity V' + sample_name))[0].T
        w_l = np.array(f_h5.get('Results/water level/water level' + sample_name)).T
        feats.append(np.flip(np.stack([u, v, w_l], axis=-1), axis=0))

        time = np.array(f_h5.get('/Time/Time' + sample_name))
        times.append(time)

    return np.stack(feats, axis=0), np.stack(times, axis=0)


def _preprocess(
        data_path: str = './KIOST_MOHID/',
        save_path: str = './load/nm_wf'
):
    os.makedirs(save_path, exist_ok=True)

    files = [data_path + f for f in os.listdir(data_path)]
    files_2km = natsort.natsorted([f for f in files if '2km' in f], reverse=False)
    files_300m = natsort.natsorted([f for f in files if '300m' in f], reverse=False)

    res_2km = process_map(get_feats_from_file, files_2km, max_workers=cpu_count(), chunksize=1)
    feats_2km = np.concatenate([r[0] for r in res_2km], axis=0)
    times_2km = np.concatenate([r[1] for r in res_2km], axis=0)
    del res_2km


    train_idx = np.flatnonzero(times_2km[:, 0] < 2022)
    val_idx = np.flatnonzero((times_2km[:, 1] % 2 == 1) & (times_2km[:, 0] == 2022))
    test_idx = np.flatnonzero((times_2km[:, 1] % 2 == 0) & (times_2km[:, 0] == 2022))
    test_idx = test_idx[:-9]

    res_300m = process_map(get_feats_from_file, files_300m, max_workers=cpu_count(), chunksize=1)
    feats_300m = np.concatenate([r[0] for r in res_300m], axis=0)
    del res_300m

    assert feats_2km.shape[0] == feats_300m.shape[0], 'Number of samples must be equal'

    # 1 for non-masked, 0 for masked. If value contains under -1e+5, it is masked.
    mask = np.where(np.where(feats_300m < -1e+5, True, False).any(axis=(0, 3)) == 1, 0, 1)[np.newaxis, ..., np.newaxis]

    # Train/Val/Test Split
    feats_2km_train = feats_2km[train_idx]
    feats_2km_val = feats_2km[val_idx]
    feats_2km_test = feats_2km[test_idx]
    del feats_2km

    feats_300m_train = feats_300m[train_idx]
    feats_300m_val = feats_300m[val_idx]
    feats_300m_test = feats_300m[test_idx]
    del feats_300m

    np.save(save_path + '/train_lr.npy', feats_2km_train)
    np.save(save_path + '/train_hr.npy', feats_300m_train)
    np.save(save_path + '/val_lr.npy', feats_2km_val)
    np.save(save_path + '/val_hr.npy', feats_300m_val)
    np.save(save_path + '/test_lr.npy', feats_2km_test)
    np.save(save_path + '/test_hr.npy', feats_300m_test)
    np.save(save_path + '/mask.npy', mask)

    files_obs = [data_path + f for f in os.listdir(data_path) if '2022년' in f]
    files_obs = natsort.natsorted(files_obs, reverse=False)

    files_obs = pd.concat([
        pd.read_csv(f, sep='\t', skiprows=3, encoding='cp949', na_values='-', parse_dates=["관측시간"]) for f in files_obs
    ])

    files_obs['관측시간'] = files_obs['관측시간'] - pd.Timedelta(hours=9) 

    files_obs = files_obs[files_obs['관측시간'].dt.year == 2022]
    files_obs = files_obs[files_obs['관측시간'].dt.minute == 0]
    files_obs = files_obs[files_obs['관측시간'].dt.second == 0]
    files_obs = files_obs[files_obs['관측시간'].dt.month % 2 == 0]

    files_obs = files_obs.sort_values(by='관측시간')

    water_speed = files_obs['유속(cm/s)'].values.astype(np.float64) / 100. 
    water_degree = 270. - files_obs['유향(deg)'].values.astype(np.float64)   
    water_degree = np.where(water_degree < 0., water_degree + 360, water_degree)

    radian = np.deg2rad(water_degree)
    u = -water_speed * np.cos(radian)
    v = -water_speed * np.sin(radian)

    np.save(save_path + '/obs_u.npy', u)
    np.save(save_path + '/obs_v.npy', v)

    # Validity Check
    assert len(test_idx) == len(u) == len(v), 'Number of samples must be equal'
    times_2km = times_2km[test_idx]

    def extract_time(time):
        return datetime(int(time[0]), int(time[1]), int(time[2]), int(time[3]), int(time[4]), int(time[5]))

    times_2km = np.array(list(map(extract_time, times_2km)))
    assert np.all(files_obs['관측시간'] == times_2km), 'Time must be equal'


if __name__ == "__main__":
    _preprocess()
