import os
import glob
import random

import numpy as np
import torch
from einops import rearrange
from torch import nn

from trRNA2.utils_3d.constants import ATOM_NUM_MAX, RESD_NAMES, ATOM_NAMES_PER_RESD


def get_clusters(clust_file, npz_pth, ext='npz'):
    all_lst = {}
    current_lst = []
    rm_id = []
    for line in open(clust_file):
        if line.startswith('>Cluster'):
            if current_lst:
                all_lst[clstr_id] = current_lst
            clstr_id = int(line.split()[1])
            current_lst = []
        else:
            pid = line.split()[2].split('>')[1].split('.')[0]
            if not os.path.isfile(f'{npz_pth}/{pid}.{ext}'):
                alt_lst = glob.glob(f'{npz_pth}/{pid}*.npz')
                if len(alt_lst) == 0:
                    print('miss', pid)
                    continue
                else:
                    npz_file = random.choice(alt_lst)
                    pid = npz_file.split('/')[-1].split('.')[0]
            current_lst.append(pid)
    train_lst = [all_lst[idx] for idx in all_lst if idx not in rm_id]
    return train_lst


def train_val_split(all_lst, args):
    os.system(
        f'{args.cdhit_pth}/cd-hit-est '
        f'-i {args.output_dir}/all.fasta '
        f'-o {args.output_dir}/all_c80_s80.fasta '
        f'-c 0.8 -s 0.8 '
        f'>>{args.output_dir}/data.log 2>&1')
    clstrs_80 = get_clusters(f'{args.output_dir}/all_c80_s80.fasta.clstr', args.train_npz_dir)

    filtered = [clstr for clstr in clstrs_80 if len(clstr) <= 10]
    sampled_clstrs = random.sample(filtered, 30)
    val_lst = []
    for clstr in sampled_clstrs:
        val_lst.extend(clstr)

    train_lst = sorted(list(set(all_lst) - set(val_lst)))
    with open(f'{args.output_dir}/train.lst', 'w') as flst:
        with open(f'{args.output_dir}/train.fasta', 'w') as fseq:
            flst.write('\n'.join(train_lst))
            for pid in train_lst:
                fseq.write(open(f'{args.train_fasta_dir}/{pid}.fasta').read().strip() + '\n')
    os.system(
        f'{args.cdhit_pth}/cd-hit-est '
        f'-i {args.output_dir}/train.fasta '
        f'-o {args.output_dir}/train_c100_s100.fasta '
        f'-c 1 -s 1 '
        f'>>{args.output_dir}/data.log 2>&1')
    train_clstrs = get_clusters(f'{args.output_dir}/train_c100_s100.fasta.clstr', args.train_npz_dir)
    return train_lst, val_lst, train_clstrs


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def FAPEloss(x, frame, x_true, frame_true, mask=None, mask2d=None, cmask=None, eps=1e-4, Z=10, d_clamp=10):
    B, n, L, d = x.size()
    #
    if mask is None: mask = torch.ones((B, L), device=x.device)
    if mask2d is None:
        mask2d = 1
    else:
        mask2d = mask2d.unsqueeze(-1).unsqueeze(-1)
    if cmask is None: cmask = torch.ones((B, n, L), device=x.device)
    mask = mask.unsqueeze(1)  # b,1,L
    mask = mask * cmask
    valid_mask1 = 1 - torch.isnan(frame_true[0][:, :, :, -1, -1]).long()  # (b, m, L)
    valid_mask2 = 1 - torch.isnan(x_true[..., 0]).long()  # (b, n, L)
    valid_mask2d = torch.einsum('bmi,bnj->bijmn', valid_mask1, valid_mask2 * mask)
    mask2d = valid_mask2d * mask2d

    R, t = frame
    R_true, t_true = frame_true
    R_true = torch.where(torch.isnan(R_true), torch.tensor(0).to(x.device), R_true)
    t_true = torch.where(torch.isnan(t_true), torch.tensor(0).to(x.device), t_true)
    x_true = torch.where(torch.isnan(x_true), torch.tensor(0).to(x.device), x_true)

    x_ij = torch.einsum('bmilk, bmnijl->bijmnk', R,
                        x[:, None, :, None, :, :] - t[:, :, None, :, None, :])  # bnjk-bmik
    x_ij_true = torch.einsum('bmilk, bmnijl->bijmnk', R_true,
                             x_true[:, None, :, None, :, :] - t_true[:, :, None, :, None, :])
    dij = torch.sum((x_ij - x_ij_true) ** 2, dim=-1) + eps

    dij = torch.clamp(dij ** .5, max=d_clamp) / Z
    dims_to_sum = tuple(range(1, dij.ndim))
    num_mask = torch.sum(torch.ones_like(dij) * mask2d, dim=dims_to_sum)
    return torch.sum(dij * mask2d, dim=dims_to_sum) / num_mask


def torsionAngleLoss(pred_angls, native_angls):
    angls_norm = pred_angls.norm(dim=-1, keepdim=True)
    normed_angls = pred_angls / angls_norm
    nan_mask = torch.isnan(native_angls[..., 0])  # b,l,n
    _native_angls = native_angls[~nan_mask]
    _normed_angls = normed_angls[:, ~nan_mask]
    Ltorsion = torch.nanmean(torch.sum((_normed_angls - _native_angls[None]) ** 2, dim=-1))
    Langlenorm = (torch.abs(angls_norm - 1)).mean()
    return Ltorsion + 0.02 * Langlenorm


def calc_atmwise_dist(cords_allatm):
    """
    :param cords_allatm: b n l 3
    :return:
    """
    cords_allatm_merged = rearrange(cords_allatm, 'b n l d->b (n l) d')
    atm_pairwise_dist = torch.cdist(cords_allatm_merged, cords_allatm_merged)
    return atm_pairwise_dist


def calc_clash_loss(outputs, native_feats=None):
    if isinstance(outputs['cords_allatm_mask'], list):
        cords_allatm_mask = outputs['cords_allatm_mask'][-1]
    else:
        cords_allatm_mask = outputs['cords_allatm_mask']
    if isinstance(outputs['cords_allatm'], list):
        cords_allatm = outputs['cords_allatm'][-1]
    else:
        cords_allatm = outputs['cords_allatm']
    cmask_allatm_merged = rearrange(cords_allatm_mask, 'b n l->b (n l)')
    cmask_allatm_merged2d = (cmask_allatm_merged[:, None] * cmask_allatm_merged[:, :, None]).bool()

    atm_pairwise_dist = calc_atmwise_dist(cords_allatm)
    clash_mask = cmask_allatm_merged2d

    if native_feats is not None:
        native_atm_pairwise_dist = calc_atmwise_dist(native_feats['cords_allatm'])
        _native_atm_pairwise_dist = torch.where(torch.isnan(native_atm_pairwise_dist), 100, native_atm_pairwise_dist)
        clash_dist = torch.minimum(_native_atm_pairwise_dist - 0.1, 2 * torch.ones_like(_native_atm_pairwise_dist))
        clash_loss = nn.ReLU()(clash_dist[clash_mask] - atm_pairwise_dist[clash_mask])
    else:
        clash_loss = nn.ReLU()(2 - atm_pairwise_dist[clash_mask])

    clash_loss = clash_loss.sum() / cmask_allatm_merged.sum()

    return clash_loss


def clash_ss3d_loss(outputs, native_feats=None, native_ss=None, tol_ss3d=3):
    if isinstance(outputs['cords_allatm'], list):
        cords_allatm = outputs['cords_allatm'][-1]
    else:
        cords_allatm = outputs['cords_allatm']

    atm_pairwise_dist = calc_atmwise_dist(cords_allatm)

    clash_loss = calc_clash_loss(outputs, native_feats)

    atm_pairwise_dist = rearrange(atm_pairwise_dist, 'b (n l) (N L)->b l L n N', n=ATOM_NUM_MAX, N=ATOM_NUM_MAX)
    if native_ss is None or native_ss.max() == 0:
        ss3d_loss1, ss3d_loss2, ss3d_loss3 = 0, 0, 0
    else:
        mask_base = native_ss[:, :, :, None, None].bool()
        mask_base = mask_base.repeat(1, 1, 1, 23, 23)
        if native_feats is not None:
            native_atm_pairwise_dist = calc_atmwise_dist(native_feats['cords_allatm'])
            native_atm_pairwise_dist = rearrange(native_atm_pairwise_dist, 'b (n l) (N L)->b l L n N', n=ATOM_NUM_MAX,
                                                 N=ATOM_NUM_MAX)
            mask1 = mask_base & (native_atm_pairwise_dist < 20)
            mask2 = mask_base & (native_atm_pairwise_dist < 12)
            ss3d_loss1 = torch.abs(atm_pairwise_dist[mask1] - native_atm_pairwise_dist[mask1]).mean()
            ss3d_loss2 = torch.abs(
                nn.ReLU()(atm_pairwise_dist[mask2] - native_atm_pairwise_dist[mask2] - tol_ss3d)).mean()
            ss3d_loss3 = torch.abs(nn.ReLU()(1.5 - atm_pairwise_dist[mask2])).mean()
        else:
            ss3d_loss1 = 0
            ss3d_loss2 = torch.abs(nn.ReLU()(atm_pairwise_dist[mask_base] - 10 - tol_ss3d)).mean()
            ss3d_loss3 = torch.abs(nn.ReLU()(1.5 - atm_pairwise_dist[mask_base])).mean()

    return clash_loss, ss3d_loss1, ss3d_loss2 + ss3d_loss3


def bond_loss(cords_allatm, seq, res_id, eps=1e-6):
    def cosangle(A, B, C):
        AB = A - B
        BC = C - B
        ABn = torch.sqrt(torch.sum(torch.square(AB), dim=-1) + eps)
        BCn = torch.sqrt(torch.sum(torch.square(BC), dim=-1) + eps)
        return torch.clamp(torch.sum(AB * BC, dim=-1) / (ABn * BCn), -0.999, 0.999)

    B, _, L, _ = cords_allatm.size()
    # get P,OP1,OP2,O3' cords
    p_mask = torch.zeros_like(cords_allatm)  # mask indicating the P atom
    o3_mask = torch.zeros_like(cords_allatm)  # mask indicating the O3' atom
    o5_mask = torch.zeros_like(cords_allatm)  # mask indicating the O3' atom
    op1_mask = torch.zeros_like(cords_allatm)  # mask indicating the OP1 atom
    op2_mask = torch.zeros_like(cords_allatm)  # mask indicating the OP2 atom
    c3_mask = torch.zeros_like(cords_allatm)  # mask indicating the OP2 atom
    for resd_name in RESD_NAMES:
        indices = [ind for ind, aa in enumerate(seq) if aa == resd_name]
        for idx_atom, atom_name in enumerate(ATOM_NAMES_PER_RESD[resd_name]):
            if atom_name == 'P':
                p_mask[:, idx_atom, indices] = 1
            if atom_name == "O3'":
                o3_mask[:, idx_atom, indices] = 1
            if atom_name == "O5'":
                o5_mask[:, idx_atom, indices] = 1
            if atom_name == "OP1":
                op1_mask[:, idx_atom, indices] = 1
            if atom_name == "OP2":
                op2_mask[:, idx_atom, indices] = 1
            if atom_name == "C3'":
                c3_mask[:, idx_atom, indices] = 1
    p_cords_before = (cords_allatm * p_mask).sum(dim=1)[:, :-1, :]
    p_cords = (cords_allatm * p_mask).sum(dim=1)[:, 1:, :]
    o3_cords = (cords_allatm * o3_mask).sum(dim=1)[:, :-1, :]
    c3_cords = (cords_allatm * c3_mask).sum(dim=1)[:, :-1, :]

    # O3'-P bond length loss
    op_bond = torch.norm(p_cords - o3_cords, dim=-1)
    if res_id is not None:
        bond_mask = (torch.diff(res_id, dim=-1) == 1).to(op_bond.device)
    else:
        bond_mask = torch.ones_like(op_bond, device=op_bond.device).bool()
    op_loss = torch.clamp(torch.abs(op_bond - 1.607) - 0.3, min=0.0)
    op_loss = (bond_mask * op_loss).sum() / (bond_mask.sum() + eps)

    # P-P distance loss
    pp_bond = torch.norm(p_cords - p_cords_before, dim=-1)
    pp_loss = torch.clamp(torch.abs(pp_bond - 5.9) - 2, min=0.0)
    pp_loss = (bond_mask * pp_loss).sum() / (bond_mask.sum() + eps)

    # bond angle loss
    bang_OPC_pred = cosangle(c3_cords, o3_cords, p_cords).reshape(B, L - 1)
    OPC_loss = torch.clamp(torch.abs(bang_OPC_pred - (-0.497)) - 0.1, min=0.0)
    OPC_loss = (bond_mask * OPC_loss).sum() / (bond_mask.sum() + eps)

    return 2 * op_loss + 2 * pp_loss + OPC_loss


def cross_entropy_binary(pred, real, eps=1e-6, mask2d=None):
    """mask2d:1,L,L  bool"""
    pred = pred.squeeze()
    real = real.squeeze()
    assert pred.size() == real.size()
    nan_mask = torch.isnan(real)
    _real = real[~nan_mask]
    _pred = pred[~nan_mask]
    loss_mat = -((_real * torch.log(_pred + eps)) + ((1 - _real) * torch.log(1 - _pred + eps)))

    if not (loss_mat > -1e-4).all():
        return 0
    if mask2d is not None:
        _mask2d = mask2d.squeeze()[~nan_mask].float()
        loss = (loss_mat * _mask2d).sum() / (_mask2d > 0).sum()
    else:
        loss = loss_mat.mean()
    return loss


def cross_entropy(pred, real, eps=1e-6, mask2d=None):
    pred = pred.squeeze()
    real = real.squeeze()
    assert pred.size() == real.size()
    nanidx = torch.isnan(real[..., 0])
    _real = real[~nanidx]
    _pred = pred[~nanidx]
    loss_mat = -(_real * torch.log(_pred + eps)).sum(dim=-1)
    if mask2d is not None:
        _mask2d = mask2d.squeeze()[~nanidx].float()
        loss = (loss_mat * _mask2d).sum() / (_mask2d > 0).sum()
    else:
        loss = loss_mat.mean()
    return loss


def cont_acc_binary(pred, real, frac=1, sep=12):
    nc = real.shape[0]
    if len(pred.shape) == 3:
        assert pred.shape[-1] == 1
        pred = pred[..., 0]

    if nc <= sep or len(pred.shape) < 2: return np.nan
    N = int(nc / frac)
    w = np.triu(pred, k=sep)
    top = np.argpartition(-w.ravel(), N)[:N]
    ngood = (np.ravel(real)[top] == 1).sum()
    return ngood / int(nc / frac)


def calculate_lddt(decoy, native, rad=30, cutoffs=[1, 2, 4, 6], mask2d=None):
    """
    :param decoy: (L,3)
    :param native: (L,3)
    :return:
    """
    if len(decoy.shape) == 3:
        decoy = decoy[0]
    if len(native.shape) == 3:
        native = native[0]
    length = decoy.shape[0]
    pred_dist = ((decoy[None, :, :] - decoy[:, None, :]) ** 2).sum(dim=-1) ** .5
    native_dist = ((native[None, :, :] - native[:, None, :]) ** 2).sum(dim=-1) ** .5

    def helper(t):
        good_mask = dist_ae < t
        all_mask = (sep_mask & dist_mask)
        all_good_mask = (all_mask & good_mask)
        local_lddt = all_good_mask.sum(dim=0) / all_mask.sum(dim=0)
        global_lddt = local_lddt.mean()
        return local_lddt

    dist_ae = abs(pred_dist - native_dist)
    i, j = np.meshgrid(np.arange(length), np.arange(length))
    sep_mask = torch.tensor(i != j, device=pred_dist.device).bool()
    dist_mask = (native_dist < rad)
    if mask2d is not None: dist_mask &= mask2d.squeeze()
    if hasattr(torch, 'nanmean'):
        local_lddt = torch.nanmean(torch.stack([helper(t) for t in cutoffs]), dim=0)
        global_lddt = torch.nanmean(local_lddt)
    else:
        local_lddt = torch.from_numpy(np.nanmean([helper(t).detach().cpu().numpy() for t in cutoffs], axis=0)).to(
            decoy.device)
    return local_lddt


def get_rotation(x, x_true):
    device = x.device
    true_points = torch.clone(x_true.squeeze(0).float())
    pred_points = torch.clone(x.squeeze(0).float())

    nan_idx = torch.isnan(true_points[:, 0])
    true_points = true_points[~nan_idx]
    pred_points = pred_points[~nan_idx]

    tt = true_points.mean(dim=0)
    tp = pred_points.mean(dim=0)
    true_points = true_points - tt[None]
    pred_points = pred_points - tp[None]
    L = true_points.size(0)

    h = pred_points.T @ true_points
    try:
        u, s, v = torch.svd(h, some=False)
    except Exception as e:
        raise e
    u = u.float()

    d = torch.linalg.det((v @ u.T).float())
    e = nn.parameter.Parameter(torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, d]]).to(device), requires_grad=False)

    r = v @ e @ u.T
    return r, tp, tt


def cRMSD(x, x_true, x_base=None, x_base_true=None):
    if x_base is None: x_base = x
    if x_base_true is None: x_base_true = x_true
    r, tp, tt = get_rotation(x_base, x_base_true)

    true_points = torch.clone(x_true.squeeze(0).float())
    pred_points = torch.clone(x.squeeze(0).float())

    nan_idx = torch.isnan(true_points[:, 0])
    _true_points = true_points[~nan_idx]
    _pred_points = pred_points[~nan_idx]
    _true_points = _true_points - tt[None]
    _pred_points = _pred_points - tp[None]

    L = _true_points.size(0)
    rotated = torch.einsum('ij,lj->li', r, _pred_points)
    dist_mat = nn.PairwiseDistance(p=2)(rotated, _true_points) ** 2
    rmsd = (dist_mat.sum() / L) ** .5
    return rmsd


def unparse_a3m(msa):
    l = 'AUCG-'

    def num_to_name(num):
        return l[num]

    # print([([(num) for num in seq]) for seq in msa])
    return [''.join([num_to_name(num) for num in seq]) for seq in msa]


def save_model(model, pth):
    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    if torch_version < 1.4:
        torch.save(model, pth)
    else:
        # _use_new_zipfile_serialization=False for old version compatibility
        torch.save(model, pth, _use_new_zipfile_serialization=False)


def load_checkpoint(ckpt_pth, model, optimizer=None, scheduler=None, opt_devices=None, init_lr=None, device='cuda'):
    out_dict = {}
    if model is not None:
        model_CKPT = torch.load(ckpt_pth, map_location=device)
        model.load_state_dict(model_CKPT['state_dict'])
        model.train()
        out_dict['model'] = model.to(device)

    if scheduler is not None:
        scheduler.load_state_dict(model_CKPT['scheduler'])
        out_dict['scheduler'] = scheduler
    if optimizer is not None:
        optimizer.load_state_dict(model_CKPT['optimizer'])
        if init_lr is not None:
            for pg in optimizer.param_groups:
                pg['lr'] = init_lr
        if opt_devices is not None:
            for ind, state in enumerate(optimizer.state.values()):
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        try:
                            state[k] = v.to(device)
                        except KeyError as e:
                            continue
        out_dict['optimizer'] = optimizer

    return out_dict
