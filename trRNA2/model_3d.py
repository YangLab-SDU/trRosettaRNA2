from collections import defaultdict

import torch
from einops import rearrange
from torch import nn

from .InitStrGenerator import InitStr_Network
from .RNAformer import RNAformer
from .utils import *
from .utils_3d.rigid_utils import Rigid
from .structure_module import StructureModule


class InputEmbedder(nn.Module):
    def __init__(self, dim=48, in_dim=46, use_ss=True):
        super(InputEmbedder, self).__init__()
        in_dim = in_dim + 1 if use_ss else in_dim
        self.use_ss = use_ss
        self.bn1 = nn.InstanceNorm2d(in_dim, affine=True)
        self.elu1 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_dim, dim, 1)
        self.linear1 = nn.Sequential(
            self.bn1,
            self.elu1,
            self.conv1
        )
        self.token_emb = nn.Embedding(5, dim)

    def forward(self, msa, ss, msa_cutoff=500):
        with torch.no_grad():
            f2d = self.get_f2d(msa[0], ss)
        pair = self.linear1(f2d.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        m = self.token_emb(msa[:, :msa_cutoff, :].long())

        return {'pair': pair, 'msa': m}

    def get_f2d(self, msa, ss):
        nrow, ncol = msa.size()[-2:]
        if nrow == 1:
            msa = msa.view(nrow, ncol).repeat(2, 1)
            nrow = 2
        msa1hot = (torch.arange(5).to(msa.device) == msa[..., None].long()).float()
        w = self.reweight(msa1hot, .8)

        # 1D features
        f1d_seq = msa1hot[0, :, :4]
        f1d_pssm = self.msa2pssm(msa1hot, w)

        f1d = torch.cat([f1d_seq, f1d_pssm], dim=1)

        # 2D features
        f2d_dca = self.fast_dca(msa1hot, w)

        f2d = torch.cat([f1d[:, None, :].repeat([1, ncol, 1]),
                         f1d[None, :, :].repeat([ncol, 1, 1]),
                         f2d_dca], dim=-1)
        f2d = f2d.view([1, ncol, ncol, 26 + 4 * 5])
        if self.use_ss:
            return torch.cat([f2d, ss.view(1, ncol, ncol, 1)], dim=-1)
        return f2d

    @staticmethod
    def msa2pssm(msa1hot, w):
        beff = w.sum()
        f_i = (w[:, None, None] * msa1hot).sum(dim=0) / beff + 1e-9
        h_i = (-f_i * torch.log(f_i)).sum(dim=1)
        return torch.cat([f_i, h_i[:, None]], dim=1)

    @staticmethod
    def reweight(msa1hot, cutoff):
        id_min = msa1hot.size(1) * cutoff
        id_mtx = torch.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])
        id_mask = id_mtx > id_min
        w = 1.0 / id_mask.sum(dim=-1).float()
        return w

    @staticmethod
    def fast_dca(msa1hot, weights, penalty=4.5):
        nr, nc, ns = msa1hot.size()
        try:
            x = msa1hot.view(nr, nc * ns)
        except RuntimeError:
            x = msa1hot.contiguous().view(nr, nc * ns)
        num_points = weights.sum() - torch.sqrt(weights.mean())

        mean = torch.sum(x * weights[:, None], dim=0, keepdim=True) / num_points
        x = (x - mean) * torch.sqrt(weights[:, None])
        cov = torch.matmul(x.permute(1, 0), x) / num_points

        cov_reg = cov + torch.eye(nc * ns).to(msa1hot.device) * penalty / torch.sqrt(weights.sum())
        inv_cov = torch.inverse(cov_reg)

        x1 = inv_cov.view(nc, ns, nc, ns)
        x2 = x1.permute(0, 2, 1, 3)
        features = x2.reshape(nc, nc, ns * ns)

        x3 = torch.sqrt((x1[:, :-1, :, :-1] ** 2).sum((1, 3))) * (1 - torch.eye(nc).to(msa1hot.device))
        apc = x3.sum(dim=0, keepdim=True) * x3.sum(dim=1, keepdim=True) / x3.sum()
        contacts = (x3 - apc) * (1 - torch.eye(nc).to(msa1hot.device))

        return torch.cat([features, contacts[:, :, None]], dim=2)


class RecyclingEmbedder(nn.Module):
    def __init__(self, dim=48):
        super(RecyclingEmbedder, self).__init__()
        self.linear = nn.Linear(38, dim)
        self.norm_pair = nn.LayerNorm(dim)
        self.norm_msa = nn.LayerNorm(dim)

    def forward(self, reprs_prev, x=None):
        if x is None:
            x = reprs_prev['x']
        d = torch.cdist(x, x)
        d = one_hot(d)
        d = self.linear(d)
        pair = self.norm_pair(reprs_prev['pair']) + d
        single = self.norm_msa(reprs_prev['single'])
        return single, pair


class Distogram(nn.Module):
    def __init__(self, dim=48):
        super(Distogram, self).__init__()
        self.out_elu_2d = nn.Sequential(
            nn.InstanceNorm2d(dim, affine=True),
            nn.ELU(inplace=True)
        )
        self.fc_2d = nn.ModuleDict(
            {
                'distance': nn.ModuleDict(
                    dict((a,
                          nn.Sequential(
                              Symm('b i j d->b j i d'),
                              nn.Linear(dim, n_bins['inter_labels']['distance'])))
                         for a in obj['inter_labels']['distance']),
                ),
                'contact': nn.ModuleDict(
                    dict(
                        (a, nn.Sequential(
                            Symm('b i j d->b j i d'),
                            nn.Linear(dim, n_bins['inter_labels']['contact'])))
                        for a in obj['inter_labels']['contact'])
                ),
            }
        )

    def forward(self, pair_repr):
        pair_repr = rearrange(self.out_elu_2d(rearrange(pair_repr, 'b i j d->b d i j')), 'b d i j->b i j d')
        pred_dict = {'inter_labels': defaultdict(dict), 'intra_labels': defaultdict(dict)}
        for k in obj['inter_labels']:
            if k != 'contact':
                for a in obj['inter_labels'][k]:
                    pred_dict['inter_labels'][k][a] = self.fc_2d[k][a](pair_repr).softmax(-1).squeeze(0)

            else:
                for a in obj['inter_labels'][k]:
                    pred_dict['inter_labels'][k][a] = self.fc_2d[k][a](pair_repr).sigmoid().squeeze()

        return pred_dict


class Folding(nn.Module):
    def __init__(self, dim_2d=48, layers_2d=12, config={}):
        super(Folding, self).__init__()

        self.config = config
        self.input_embedder = InputEmbedder(dim=dim_2d, use_ss=config['use_ss'])
        self.recycle_embedder = RecyclingEmbedder(dim=dim_2d)
        self.net2d = RNAformer(dim=dim_2d,
                               depth=layers_2d,
                               msa_tie_row_attn=config['RNAformer']['msa_tie_row_attn'],
                               attn_dropout=config['RNAformer']['dropout_rate_attn'],
                               ff_dropout=config['RNAformer']['dropout_rate_ff'])
        structure_module_config = config['structure_module']
        dim_3d = structure_module_config['c_z']
        self.to_dist = Distogram(dim_2d)
        if config['init_str'] == 'nn':
            self.init_str = InitStr_Network(
                node_dim_in=dim_2d, edge_dim_in=dim_2d,
                node_dim_hidden=dim_3d, edge_dim_hidden=dim_3d,
                use_ss='ss3D' in config and config['ss3D'],
                aa_types=5,
                nblocks=2, dropout=.3, out_fmt='quats')
        self.structure_module = StructureModule(
            ss_usage='multiply' if 'ss_usage' not in structure_module_config else structure_module_config[
                'ss_usage'],
            **structure_module_config
        )
        self.to_ss = nn.Sequential(
            nn.LayerNorm(dim_3d),
            nn.Linear(dim_3d, dim_3d),
            Symm('b i j d->b j i d'),
            nn.Linear(dim_3d, dim_3d),
            nn.ReLU(),
            nn.LayerNorm(dim_3d),
            nn.Dropout(.1),
            nn.Linear(dim_3d, 1)
        )
        self.to_plddt = nn.Sequential(
            nn.LayerNorm(dim_3d),
            nn.Linear(dim_3d, dim_3d),
            nn.ReLU(),
            nn.Linear(dim_3d, dim_3d),
            nn.ReLU(),
            nn.Linear(dim_3d, 50),
        )

    def forward(self, raw_seq, msa, ss, res_id=None, num_recycle=3, msa_cutoff=500, return_mid=False,
                return_attn=False, config={}):
        reprs_prev = None
        outputs_all = {}
        for c in range(1 + num_recycle):
            with torch.set_grad_enabled(False):
                with torch.cuda.amp.autocast(enabled=len(raw_seq) > 300):
                    reprs = self.input_embedder(msa, ss, msa_cutoff=msa_cutoff)
                    if reprs_prev is None:
                        reprs_prev = {
                            'pair': torch.zeros_like(reprs['pair']),
                            'single': torch.zeros_like(reprs['msa'][:, 0]),
                            'x': torch.zeros(list(reprs['pair'].shape[:2]) + [3], device=reprs['pair'].device),
                        }
                    t = reprs_prev['x']
                    rec_msa, rec_pair = self.recycle_embedder(reprs_prev, t)
                    reprs['msa'][:, 0] = reprs['msa'][:, 0] + rec_msa
                    reprs['pair'] = reprs['pair'] + rec_pair
                    out = self.net2d(reprs['pair'], msa_emb=reprs['msa'], return_msa=True, res_id=res_id,
                                     preprocess=False, return_attn=c == num_recycle)
                    torch.cuda.empty_cache()
                    if c != num_recycle:
                        pair_repr, msa_repr = out
                    else:
                        pair_repr, msa_repr, attn_maps = out
                reprs = {'msa': msa_repr, 'pair': pair_repr}

                if ('ss3D' in config and config['ss3D']):
                    input_ss = ss.float()
                else:
                    input_ss = None

                if config['init_str'] == 'nn':
                    seq1hot = (torch.arange(5, device=msa.device) == msa[0, 0:1, :, None]).float()
                    R, t, quats, repr_gt = self.init_str(seq1hot, reprs['pair'], idx=res_id, msa=reprs['msa'],
                                                         ss=input_ss,
                                                         reweight=True, return_repr=True)
                    _reprs = {"single": repr_gt['seq'], "pair": repr_gt['pair']}
                    if 'divide' in config and config['divide']:
                        tensor7 = torch.cat([quats, t / config['structure_module']['trans_scale_factor']], dim=-1)
                    else:
                        tensor7 = torch.cat([quats, t], dim=-1)
                    rigids = Rigid.from_tensor_7(tensor7, normalize_quats=True)
                else:
                    _reprs = {"single": msa_repr[:, 0], "pair": pair_repr}
                    rigids = None

                outputs = self.structure_module.forward(raw_seq, _reprs, ss=input_ss, rigids=rigids,
                                                        return_mid=c == num_recycle or return_mid)

                reprs_prev = {
                    'single': msa_repr[..., 0, :, :].detach(),
                    'pair': pair_repr.detach(),
                    "x": outputs["cords_c1'"][-1].detach(),
                }
                if return_mid or c == num_recycle:
                    outputs['geoms'] = self.to_dist(pair_repr)
                    outputs['cords_allatm'] = outputs['cord_tns_pred'][-1]
                    outputs['cords_allatm_mask'] = outputs['cmask'][-1]
                    outputs['atm_name'] = outputs['atm_name'][-1]
                    outputs["cords_c1'"] = torch.stack(outputs["cords_c1'"], dim=0)

                    plddt_prob = self.to_plddt(outputs['single']).softmax(-1)
                    plddt = torch.einsum('lbik,k->lbi', plddt_prob,
                                         torch.arange(0.01, 1.01, 0.02, device=plddt_prob.device))
                    outputs['plddt_prob'] = plddt_prob
                    outputs['plddt'] = plddt

                    outputs['ss'] = self.to_ss(pair_repr).sigmoid()

                    rots = []
                    tsls = []
                    if config['init_str'] == 'nn':
                        rots.append(R)
                        tsls.append(t)
                    for frames in outputs['frames']:
                        rigid = Rigid.from_tensor_7(frames, normalize_quats=True)
                        fram = rigid.to_tensor_4x4()
                        rots.append(fram[:, :, :3, :3])
                        tsls.append(fram[:, :, :3, 3:].squeeze(-1))
                    outputs['frames'] = (torch.stack(rots, dim=0), torch.stack(tsls, dim=0))

                    outputs['frames_allatm'] = {k: (v[:, :, :3, :], v[:, :, 3, :]) for k, v in
                                                outputs['frames_allatm'].items()}
                    outputs['frames_allatm']['main'] = [arr[-1] for arr in outputs['frames']]
                    if return_attn:
                        outputs['attn_maps'] = (attn_maps)
                    outputs_all[c] = (outputs)
                    torch.cuda.empty_cache()
        return outputs_all, outputs
