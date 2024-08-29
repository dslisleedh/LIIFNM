import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils
from test_nm import test_metrics

import numpy as np

from nm_test_funcs import preprocess_nan, preprocess_normalize

import copy


def make_data_loader(spec_train, spec_valid, train_inputs: dict, val_inputs: dict):
    train_dataset = datasets.image_folder.PairedNMFolders(**train_inputs)
    train_dataset = datasets.wrappers.SRImplicitPaired(train_dataset, **spec_train['wrapper']['args'])
    val_dataset = datasets.image_folder.PairedNMFolders(**val_inputs)
    val_dataset = datasets.wrappers.SRImplicitPairedTest(val_dataset, **spec_valid['wrapper']['args'])

    train_loader = DataLoader(train_dataset, batch_size=spec_train['batch_size'],
        shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=spec_valid['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader


def make_train_data_loaders():
    train_lr = np.load('./load/nm_wf/train_lr.npy')
    train_hr = np.load('./load/nm_wf/train_hr.npy')
    val_lr = np.load('./load/nm_wf/val_lr.npy')
    val_hr = np.load('./load/nm_wf/val_hr.npy')

    # Remove erroneous data
    train_lr = train_lr[:, :, 2:, :]
    train_hr = train_hr[:, :, 12:, :]
    val_lr = val_lr[:, :, 2:, :]
    val_hr = val_hr[:, :, 12:, :]

    # Fill NaN Data in LR
    train_lr, train_hr, train_mask = preprocess_nan(train_lr, train_hr, 'train')
    val_lr, val_hr, val_mask = preprocess_nan(val_lr, val_hr, 'val')

    max_val = max(np.max(train_lr), np.max(train_hr))
    min_val = min(np.min(train_lr), np.min(train_hr))

    train_lr, train_hr = preprocess_normalize(train_lr, train_hr, min_val=min_val, max_val=max_val)
    val_lr, val_hr = preprocess_normalize(val_lr, val_hr, min_val=min_val, max_val=max_val)

    train_loader, valid_loader = make_data_loader(
        config.get('train_dataset'), config.get('val_dataset'),
        {'lr_array': train_lr, 'hr_array': train_hr, 'mask_array': train_mask},
        {'lr_array': val_lr, 'hr_array': val_hr, 'mask_array': val_mask}
    )
    return train_loader, valid_loader, min_val.tolist(), max_val.tolist()


def prepare_training(save_path):
    if config.get('resume') is not None:
        if config.get('resume_from') is not None:
            load_epoch = config.get('resume_from')
            sv_file = torch.load(os.path.join(save_path, f'epoch-{load_epoch}.pth'), map_location='cpu')
        else:
            sv_file = torch.load(os.path.join(save_path, 'epoch-last.pth'), map_location='cpu')
        model = models.make(sv_file['model']).cpu()
        model.load_state_dict(sv_file['model']['sd'])
        model.cuda()

        if config.get('use_auxiliary_train_step'):
            optimizer_fixed = utils.make_optimizer(
                model.encoder.parameters(), sv_file['optimizer_fixed'], load_sd=True)
            if config['model']['args']['encoder_spec']['args'].get('w_split_ratio', None):
                trainable_parameters = list(model.uvnet.parameters()) + list(model.wnet.parameters())
            else:
                if hasattr(model, 'imnet'):
                    trainable_parameters = list(model.imnet.parameters())
                else:
                    trainable_parameters = list(model.render.parameters())
            if config['train_variable_feat_extractor']:
                trainable_parameters += list(model.encoder.parameters())
            optimizer_liif = utils.make_optimizer(
                trainable_parameters, sv_file['optimizer_liif'], load_sd=True)

            epoch_start = sv_file['epoch'] + 1
            
            if config.get('multi_step_lr') is None:
                lr_scheduler_fixed = None
                lr_scheduler_liif = None
            else:
                lr_scheduler_fixed = MultiStepLR(optimizer_fixed, **config['multi_step_lr'])
                lr_scheduler_liif = MultiStepLR(optimizer_liif, **config['multi_step_lr'])
                
                lr_scheduler_fixed.load_state_dict(sv_file['lr_scheduler_fixed'])
                lr_scheduler_liif.load_state_dict(sv_file['lr_scheduler_liif'])
                
                print("LR Scheduler Fixed: {}".format(lr_scheduler_fixed.state_dict()))

        else:
            optimizer_fixed = None
            trainable_parameters = list(model.parameters())
            optimizer_liif = utils.make_optimizer(trainable_parameters, sv_file['optimizer_liif'], load_sd=True)
            epoch_start = sv_file['epoch'] + 1
            lr_scheduler_fixed = None
            if config.get('multi_step_lr') is None:
                lr_scheduler_liif = None
            else:
                lr_scheduler_liif = MultiStepLR(optimizer_liif, **config['multi_step_lr'])
                lr_scheduler_liif.load_state_dict(sv_file['lr_scheduler_liif'])
                
                print("LR Scheduler LIIF: {}".format(lr_scheduler_liif.state_dict()))    
        
        import gc
        del sv_file
        gc.collect()
        torch.cuda.empty_cache()
        
        print("Train resumed from epoch {}".format(epoch_start))

    else:
        if config.get('use_auxiliary_train_step'):
            model = models.make(config['model']).cuda()
            log('model: #params={}'.format(utils.compute_num_params(model, text=True)))

            optimizer_fixed = utils.make_optimizer(
                model.encoder.parameters(), config['optimizer'])

            if config.get('model').get('args').get('encoder_spec').get('args').get('w_split_ratio') not in [None, 'none', 'None', 'NONE']:
                trainable_parameters = list(model.uvnet.parameters()) + list(model.wnet.parameters())
            else:
                trainable_parameters = list(model.imnet.parameters())

            if config['train_variable_feat_extractor']:
                trainable_parameters += list(model.encoder.parameters())

            optimizer_liif = utils.make_optimizer(trainable_parameters, config['optimizer'])

            epoch_start = 1
            if config.get('multi_step_lr') is None:
                lr_scheduler_fixed = None
                lr_scheduler_liif = None
            else:
                lr_scheduler_fixed = MultiStepLR(optimizer_fixed, **config['multi_step_lr'])
                lr_scheduler_liif = MultiStepLR(optimizer_liif, **config['multi_step_lr'])
        
        else:
            model = models.make(config['model']).cuda()
            log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
            
            optimizer_fixed = None
            
            trainable_parameters = list(model.parameters())
            optimizer_liif = utils.make_optimizer(trainable_parameters, config['optimizer'])
            epoch_start = 1
            lr_scheduler_fixed = None
            if config.get('multi_step_lr') is None:
                lr_scheduler_liif = None
            else: 
                lr_scheduler_liif = MultiStepLR(optimizer_liif, **config['multi_step_lr'])

    return model, optimizer_fixed, optimizer_liif, epoch_start, lr_scheduler_fixed, lr_scheduler_liif


def train(train_loader, model, optimizer_fixed, optimizer_liif):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss_fixed = utils.Averager()
    train_loss_liif = utils.Averager()

    """
    1. Train Feature Extractor/Upscaler
    2. Train LIIF Module
    """
    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()
        inp = batch['inp']
        mask = batch['mask']
        gt = batch['gt']

        gt_fixed = batch['gt_fixed']
        if config['use_auxiliary_train_step']:
            pred_fixed = model.forward_fixed(inp)
            loss = torch.sum(torch.abs((pred_fixed * mask) - (gt_fixed * mask))) / mask.sum()  
            train_loss_fixed.add(loss.item())

            optimizer_fixed.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer_fixed.step()

            pred_fixed = None; loss = None

        pred_liif = model(inp, batch['coord'], batch['cell'])
        loss = loss_fn(pred_liif, gt)

        train_loss_liif.add(loss.item())

        optimizer_liif.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer_liif.step()

        pred_liif = None; loss = None

    return train_loss_fixed.item() if config['use_auxiliary_train_step'] else 0., train_loss_liif.item()


def main(config_, save_path):
    global config, log, writer
    config = config_

    log, writer = utils.set_save_path(save_path, remove=True if config.get('resume') is None else False)

    train_loader, valid_loader, min_val, max_val = make_train_data_loaders()

    model, optimizer_fixed, optimizer_liif, epoch_start, lr_scheduler_fixed, lr_scheduler_liif = prepare_training(save_path)
    
    print(model)
    
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = 1e18

    config['min_val'] = min_val
    config['max_val'] = max_val

    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer_liif.param_groups[0]['lr'], epoch)

        train_loss_fixed, train_loss_liif = train(train_loader, model, optimizer_fixed, optimizer_liif)
        if lr_scheduler_fixed is not None:
            lr_scheduler_fixed.step()
        if lr_scheduler_liif is not None:
            lr_scheduler_liif.step()

        log_info.append('train: loss={:.4f}'.format(train_loss_liif))
        writer.add_scalars('loss', {'train': train_loss_liif}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        sv_file = {
            'model': model_spec, 'epoch': epoch
        }
        if optimizer_fixed is not None:
            optimizer_fixed_spec = copy.deepcopy(config['optimizer'])
            optimizer_fixed_spec['sd'] = optimizer_fixed.state_dict()
            sv_file['optimizer_fixed'] = optimizer_fixed_spec
        if optimizer_liif is not None:
            optimizer_liif_spec = copy.deepcopy(config['optimizer'])
            optimizer_liif_spec['sd'] = optimizer_liif.state_dict()
            sv_file['optimizer_liif'] = optimizer_liif_spec
        if lr_scheduler_fixed is not None:
            sv_file['lr_scheduler_fixed'] = lr_scheduler_fixed.state_dict()
        if lr_scheduler_liif is not None:
            sv_file['lr_scheduler_liif'] = lr_scheduler_liif.state_dict()

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            res = test_metrics(valid_loader, model_, min_v=min_val, max_v=max_val)

            log_info.append(
                'val: fixed_uv={:.4f} / fixed_w={:.4f} / liif_uv={:.4f} / liif_w={:.4f}'.format(
                    res['mse']['fixed_uv'], res['mse']['fixed_w'], res['mse']['arb_uv'], res['mse']['arb_w']
                )
            )
            writer.add_scalars('fixed_uv', {'val': res['mse']['fixed_uv']}, epoch)
            writer.add_scalars('fixed_w', {'val': res['mse']['fixed_w']}, epoch)
            writer.add_scalars('liif_uv', {'val': res['mse']['arb_uv']}, epoch)
            writer.add_scalars('liif_w', {'val': res['mse']['arb_w']}, epoch)
            if res['mse']['arb_uv'] < max_val_v:
                max_val_v = res['mse']['arb_uv']
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
