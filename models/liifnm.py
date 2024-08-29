import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
from copy import deepcopy


@register('liif-nm')
class LIIFNM(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)  

        if imnet_spec is not None:
            if self.encoder.args.w_split_ratio not in [None, 'none', 'None', 'NONE']:
                w_ratio = eval(self.encoder.args.w_split_ratio)
                uvnet_spec = deepcopy(imnet_spec)
                uvnet_spec['out_dim'] = 2
                wnet_spec = deepcopy(imnet_spec)
                wnet_spec['out_dim'] = 1

                wnet_spec['args']['hidden_list'] = [int(hl * w_ratio) for hl in wnet_spec['args']['hidden_list']]
                uvnet_spec['args']['hidden_list'] = [
                    uv_hl - w_hl for uv_hl, w_hl in zip(
                        uvnet_spec['args']['hidden_list'], wnet_spec['args']['hidden_list'])
                ]

                uvnet_in_dim = self.encoder.n_feats_uv
                wnet_in_dim = self.encoder.n_feats_w
                if self.feat_unfold:
                    uvnet_in_dim *= 9
                    wnet_in_dim *= 9
                uvnet_in_dim += 2  
                wnet_in_dim += 2  
                if self.cell_decode:
                    uvnet_in_dim += 2
                    wnet_in_dim += 2

                self.uvnet = models.make(uvnet_spec, args={'in_dim': uvnet_in_dim, 'out_dim': 2})
                self.wnet = models.make(wnet_spec, args={'in_dim': wnet_in_dim, 'out_dim': 1})

            else:
                imnet_in_dim = self.encoder.n_feats
                if self.feat_unfold:
                    imnet_in_dim *= 9
                imnet_in_dim += 2  
                if self.cell_decode:
                    imnet_in_dim += 2
                self.imnet = models.make(imnet_spec, args={
                    'in_dim': imnet_in_dim, 'out_dim': 3
                })

        else:
            self.imnet = None

    def forward_fixed(self, inp):
        self.encoder.return_upsample = True
        return self.encoder(inp)

    def gen_feat(self, inp):
        self.encoder.return_upsample = False
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        if self.encoder.split_uv_w:
            uv, w = torch.split(self.feat, [self.encoder.n_feats_uv, self.encoder.n_feats_w], dim=1)
        else:
            feat = self.feat

        if self.encoder.split_uv_w:
            if self.uvnet is None:
                uv_ret = F.grid_sample(uv, coord.flip(-1).unsqueeze(1),
                                       mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                w_ret = F.grid_sample(w, coord.flip(-1).unsqueeze(1),
                                      mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                ret = torch.cat([uv_ret, w_ret], dim=-1)
                return ret
        else:
            if self.imnet is None:
                ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                return ret

        if self.feat_unfold:
            if self.encoder.split_uv_w:
                uv = F.unfold(uv, 3, padding=1).view(
                    uv.shape[0], uv.shape[1] * 9, uv.shape[2], uv.shape[3])
                w = F.unfold(w, 3, padding=1).view(
                    w.shape[0], w.shape[1] * 9, w.shape[2], w.shape[3])
            else:
                feat = F.unfold(feat, 3, padding=1).view(
                    feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # UV and W have same spatial shape
        feat_spatial_shape = feat.shape[-2:] if not self.encoder.split_uv_w else uv.shape[-2:]
        rx = 2 / feat_spatial_shape[0] / 2
        ry = 2 / feat_spatial_shape[1] / 2

        feat_coord = make_coord(feat_spatial_shape, flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(coord.shape[0], 2, *feat_spatial_shape)

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                if self.encoder.split_uv_w:
                    q_uv = F.grid_sample(
                        uv, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                    q_w = F.grid_sample(
                        w, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)

                else:
                    q_feat = F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat_spatial_shape[0]
                rel_coord[:, :, 1] *= feat_spatial_shape[1]

                if self.encoder.split_uv_w:
                    uv_inp = torch.cat([q_uv, rel_coord.clone()], dim=-1)
                    w_inp = torch.cat([q_w, rel_coord.clone()], dim=-1)
                else:
                    inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat_spatial_shape[0]
                    rel_cell[:, :, 1] *= feat_spatial_shape[1]
                    if self.encoder.split_uv_w:
                        uv_inp = torch.cat([uv_inp, rel_cell.clone()], dim=-1)
                        w_inp = torch.cat([w_inp, rel_cell.clone()], dim=-1)
                    else:
                        inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                if self.encoder.split_uv_w:
                    uv_pred = self.uvnet(uv_inp.view(bs * q, -1)).view(bs, q, -1)
                    w_pred = self.wnet(w_inp.view(bs * q, -1)).view(bs, q, -1)
                    pred = torch.cat([uv_pred, w_pred], dim=-1)
                else:
                    pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
