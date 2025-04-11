from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv


def get_seqsep(idx):
    '''
    Input:
        - idx: residue indices of given sequence (B,L)
    Output:
        - seqsep: sequence separation feature with sign (B, L, L, 1)
                  Sergey found that having sign in seqsep features helps a little
    '''
    seqsep = idx[:, None, :] - idx[:, :, None]
    sign = torch.sign(seqsep)
    seqsep = torch.log(torch.abs(seqsep) + 1.0)
    seqsep = torch.clamp(seqsep, 0.0, 5.5)
    seqsep = sign * seqsep
    return seqsep.unsqueeze(-1)


def make_graph(node, idx, emb):
    ''' create torch_geometric graph from Trunk outputs '''
    device = emb.device
    B, L = emb.shape[:2]

    # |i-j| <= kmin (connect sequentially adjacent residues)
    sep = idx[:, None, :] - idx[:, :, None]
    sep = sep.abs()
    b, i, j = torch.where(sep > 0)

    src = b * L + i
    tgt = b * L + j

    x = node.reshape(B * L, -1)

    G = Data(x=x,
             edge_index=torch.stack([src, tgt]),
             edge_attr=emb[b, i, j])

    return G


class UniMPBlock(nn.Module):
    '''https://arxiv.org/pdf/2009.03509.pdf'''

    def __init__(self,
                 node_dim=64,
                 edge_dim=64,
                 heads=4,
                 dropout=0.15):
        super(UniMPBlock, self).__init__()

        self.TConv = TransformerConv(node_dim, node_dim, heads, dropout=dropout, edge_dim=edge_dim)
        self.LNorm = LayerNorm(node_dim * heads)
        self.Linear = nn.Linear(node_dim * heads, node_dim)
        self.Activ = nn.ELU(inplace=True)

    # @torch.cuda.amp.autocast(enabled=True)
    def forward(self, G):
        xin, e_idx, e_attr = G.x, G.edge_index, G.edge_attr
        x = self.TConv(xin, e_idx, e_attr)
        x = self.LNorm(x)
        x = self.Linear(x)
        out = self.Activ(x + xin)
        return Data(x=out, edge_index=e_idx, edge_attr=e_attr)


class BackBoneUpdate(nn.Module):
    def __init__(self, dim):
        super(BackBoneUpdate, self).__init__()
        self.linear = nn.Linear(dim, 6)

    def forward(self, single_repr):
        batch_size, length = single_repr.size()[:2]
        quaternion = self.linear(single_repr)
        b = quaternion[..., 0]
        c = quaternion[..., 1]
        d = quaternion[..., 2]
        t = quaternion[..., 3:]
        norm = (1 + b ** 2 + c ** 2 + d ** 2) ** .5
        a = 1 / norm
        b = b / norm
        c = c / norm
        d = d / norm
        quats = torch.stack([a, b, c, d], dim=-1)
        R = torch.zeros((batch_size, length, 3, 3), device=single_repr.device)
        R[:, :, 0, 0] = a ** 2 + b ** 2 - c ** 2 - d ** 2
        R[:, :, 0, 1] = 2 * b * c - 2 * a * d
        R[:, :, 0, 2] = 2 * b * d + 2 * a * c
        R[:, :, 1, 0] = 2 * b * c + 2 * a * d
        R[:, :, 1, 1] = a ** 2 - b ** 2 + c ** 2 - d ** 2
        R[:, :, 1, 2] = 2 * c * d - 2 * a * b
        R[:, :, 2, 0] = 2 * b * d - 2 * a * c
        R[:, :, 2, 1] = 2 * c * d + 2 * a * b
        R[:, :, 2, 2] = a ** 2 - b ** 2 - c ** 2 + d ** 2

        # t_norm = torch.norm(t, dim=-1, keepdim=True)
        #
        # return R, 3.8 * t / t_norm
        return R, t, quats


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2 * (x - mean)
        x /= std
        x += self.b_2
        return x


class SequenceWeight(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(SequenceWeight, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, msa):
        B, N, L = msa.shape[:3]

        msa = msa.permute(0, 2, 1, 3)  # (B, L, N, K)
        tar_seq = msa[:, :, 0].unsqueeze(2)  # (B, L, 1, K)

        q = self.to_query(tar_seq).view(B, L, 1, self.heads, self.d_k).permute(0, 1, 3, 2,
                                                                               4).contiguous()  # (B, L, h, 1, k)
        k = self.to_key(msa).view(B, L, N, self.heads, self.d_k).permute(0, 1, 3, 4, 2).contiguous()  # (B, L, h, k, N)

        q = q * self.scale
        attn = torch.matmul(q, k)  # (B, L, h, 1, N)
        attn = F.softmax(attn, dim=-1)
        return self.dropout(attn)


class InitStr_Network(nn.Module):
    def __init__(self,
                 node_dim_in=64,
                 node_dim_hidden=64,
                 edge_dim_in=256,
                 edge_dim_hidden=64,
                 aa_types=21,
                 nheads=4,
                 nblocks=3,
                 dropout=0.1,
                 out_atoms=3,
                 out_fmt='coord',
                 split_pr=False,
                 use_ss=False
                 ):
        super(InitStr_Network, self).__init__()

        # embedding layers for node and edge features
        self.norm_node = LayerNorm(node_dim_in)
        self.norm_edge = LayerNorm(edge_dim_in)
        self.encoder_seq = SequenceWeight(node_dim_in, 1, dropout=dropout)

        # self.embed_x = nn.Sequential(nn.Linear(node_dim_in + aa_types, node_dim_hidden), nn.ELU(inplace=False))
        self.embed_x = nn.Linear(node_dim_in + aa_types, node_dim_hidden)
        # self.embed_e = nn.Sequential(nn.Linear(edge_dim_in + 1, edge_dim_hidden), nn.ELU(inplace=False))
        self.embed_e = nn.Linear(edge_dim_in + 2 if use_ss else edge_dim_in + 1, edge_dim_hidden)

        # graph transformer
        blocks = [UniMPBlock(node_dim_hidden, edge_dim_hidden, nheads, dropout) for _ in range(nblocks)]
        self.transformer = nn.Sequential(*blocks)

        # outputs
        if out_fmt == 'coord':
            self.get_xyz = nn.Linear(node_dim_hidden, 3 * out_atoms)
        elif out_fmt in ['frame', 'quats']:
            self.get_xyz = BackBoneUpdate(node_dim_hidden)
        if split_pr:
            self.get_xyz_lig = deepcopy(self.get_xyz)

        self.out_fmt = out_fmt
        self.out_atoms = out_atoms
        self.split_pr = split_pr
        self.use_ss = use_ss

    def forward(self, seq1hot, pair, ss=None, idx=None, mol_type=None, msa=None, return_repr=False, reweight=False):
        B, L, _, _ = pair.size()
        if idx is None:
            idx = torch.arange(L, device=pair.device).long()
        if not hasattr(self,'split_pr') or not self.split_pr or mol_type is None:
            mol_type = torch.zeros(L, device=pair.device).long()
        mol_type = mol_type.view(L)
        idx = idx.view(B, L)
        if msa is not None:
            msa = self.norm_node(msa)
        pair = self.norm_edge(pair)
        if msa is not None:
            N = msa.size(-3)
            if reweight:
                w_seq = self.encoder_seq(msa).reshape(B, L, 1, N).permute(0, 3, 1, 2)
                msa = w_seq * msa
            msa = msa.sum(dim=1)
            node = torch.cat((msa, seq1hot), dim=-1)
        else:
            node = seq1hot
        node = self.embed_x(node)

        seqsep = get_seqsep(idx)
        pair = torch.cat((pair, seqsep), dim=-1)
        if not hasattr(self,'use_ss'):
            self.use_ss = True
        if ss is not None and self.use_ss:
            pair = torch.cat((pair, ss if ss.size(-1)==1 else ss[..., None]), dim=-1)

        pair = self.embed_e(pair)

        G = make_graph(nn.ELU()(node), idx, nn.ELU()(pair))
        Gout = self.transformer(G)

        if hasattr(self,'split_pr'):
            split_pr = self.split_pr and (mol_type == 1).sum() > 0
        else:
            split_pr=False
        if self.out_fmt == 'coord':
            if split_pr:
                xyz_rna = self.get_xyz(Gout.x.view(B, L, -1)[:, mol_type == 0])
                xyz_prot = self.get_xyz_lig(Gout.x.view(B, L, -1)[:, mol_type == 1])
                xyz = torch.cat([xyz_rna, xyz_prot], dim=1)
            else:
                xyz = self.get_xyz(Gout.x)
            if return_repr:
                return xyz.view(B, L, self.out_atoms, 3), {'seq': node, 'pair': pair}
            return xyz.view(B, L, self.out_atoms, 3)  # torch.cat([xyz,node_emb],dim=-1)
        else:
            if split_pr:
                R_rna, t_rna, quaternion_rna = self.get_xyz(Gout.x.view(B, L, -1)[:, mol_type == 0])
                R_prot, t_prot, quaternion_prot = self.get_xyz_lig(Gout.x.view(B, L, -1)[:, mol_type == 1])
                R = torch.cat([R_rna, R_prot], dim=1)
                t = torch.cat([t_rna, t_prot], dim=1)
                quaternion = torch.cat([quaternion_rna, quaternion_prot], dim=1)
            else:
                R, t, quaternion = self.get_xyz(Gout.x.view(B, L, -1))
            if self.out_fmt == 'frame':
                if return_repr:
                    return R, t, {'seq': node, 'pair': pair}
                return R, t  # torch.cat([xyz,node_emb],dim=-1)
            else:
                if return_repr:
                    return R, t, quaternion, {'seq': node, 'pair': pair}
                return R, t, quaternion  # torch.cat([xyz,node_emb],dim=-1)
