# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np
from collections import defaultdict
from .constants import RNA_CONSTANTS, ANGL_INFOS_PER_RESD, TRANS_DICT_PER_RESD
from .rigid_utils import Rigid, calc_rot_tsl, calc_angl_rot_tsl, merge_rot_tsl


def calc_angls(coords, seq, angl_infos=ANGL_INFOS_PER_RESD):
    L = len(coords["P"].squeeze(0))

    angls = {f'angl_{i}': torch.nan * torch.ones((L, 2), device=coords["P"].device) for i in range(4)}
    for res, angl_info in angl_infos.items():
        indices = [ind for ind, aa in enumerate(seq) if aa == res]
        for angl, _, atms in angl_info:
            atm1_prev, atm2_prev, atm3_prev = atms[:3]
            atm1, atm2, atm3 = atms[-1], atms[1], atms[2]
            rot_prev, tsl_prev = calc_rot_tsl(coords[atm1_prev].squeeze(0).float()[indices],
                                              coords[atm3_prev].squeeze(0).float()[indices],
                                              coords[atm2_prev].squeeze(0).float()[indices])
            rot_cur, tsl_cur = calc_rot_tsl(coords[atm1].squeeze(0).float()[indices],
                                            coords[atm3].squeeze(0).float()[indices],
                                            coords[atm2].squeeze(0).float()[indices])

            rot_cur_to_prev = torch.einsum('...ji,...jk->...ik', rot_prev, rot_cur)
            #         print(np.round(rot_cur_to_prev,3))
            cos = rot_cur_to_prev[:, 1, 1]
            sin = rot_cur_to_prev[:, 2, 1]
            norm = (cos ** 2 + sin ** 2) ** .5
            if (torch.abs(norm - 1) > 0.1).any():
                raise ValueError(f'cos^2 + sin^2!=1')
            cos /= norm
            sin /= norm
            angls[angl][indices, 0] = cos
            angls[angl][indices, 1] = sin
    return angls


def calc_angls_from_base(coords, seq, angl_infos=ANGL_INFOS_PER_RESD, trans_dict=TRANS_DICT_PER_RESD,
                         ignore_norm_err=False):
    L = len(coords["P"].squeeze(0))
    device = coords["P"].device
    angls = {f'angl_{i}': torch.nan * torch.ones((L, 2), device=device) for i in range(4)}
    for res, angl_info in angl_infos.items():
        indices = [ind for ind, aa in enumerate(seq) if aa == res]
        fram_dict = {}
        atm1, atm2, atm3 = ["C4'", "N9", "C1'"] if res in ['A', 'G'] else ["C4'", "N1", "C1'"]
        fram_dict['main'] = calc_rot_tsl(coords[atm1].squeeze(0).float()[indices],
                                         coords[atm3].squeeze(0).float()[indices],
                                         coords[atm2].squeeze(0).float()[indices])
        for angl, _, atms in angl_info:
            if angl in ['omega', 'phi', 'angl_0', 'angl_1']:  # 上一级是主链
                angl_prev = 'main'
            else:
                angl_prev = 'angl_%d' % (int(angl[-1]) - 1)  # 上一级是另一个侧链坐标系
            atm1, atm2, atm3 = atms[-1], atms[1], atms[2]
            rot_curr, tsl_curr = calc_rot_tsl(coords[atm1].squeeze(0).float()[indices],
                                              coords[atm3].squeeze(0).float()[indices],
                                              coords[atm2].squeeze(0).float()[indices])
            fram_dict[angl] = (rot_curr, tsl_curr)

            rot_prev, tsl_prev = fram_dict[angl_prev]
            rot_base, tsl_vec_base = trans_dict[res]['%s-%s' % (angl, angl_prev)]
            rot_base = torch.from_numpy(rot_base).float().to(device)  # 到上一个坐标系的转换
            tsl_base = torch.from_numpy(tsl_vec_base).float().to(device)

            rot_base, tsl_base = merge_rot_tsl(rot_prev, tsl_prev, rot_base, tsl_base)
            rot_addi = torch.einsum('...ji,...jk->...ik', rot_base, rot_curr)

            cos = rot_addi[:, 1, 1]
            sin = rot_addi[:, 2, 1]
            norm = (cos ** 2 + sin ** 2) ** .5
            if not ignore_norm_err and (torch.abs(norm - 1) > 0.1).any():
                raise ValueError(f'cos^2 + sin^2!=1')
            cos /= norm
            sin /= norm
            angls[angl][indices, 0] = cos
            angls[angl][indices, 1] = sin
    return angls


class RNAConverter():
    """RNA Structure Converter."""

    def __init__(self):
        """"""

        self.eps = 1e-4
        self.__init()

    def __init(self):
        """"""

        self.cord_dict = defaultdict(dict)
        for resd_name in RNA_CONSTANTS.RESD_NAMES:
            for atom_name, _, cord_vals in RNA_CONSTANTS.ATOM_INFOS_PER_RESD[resd_name]:
                self.cord_dict[resd_name][atom_name] = torch.tensor(cord_vals, dtype=torch.float32)

        # trans_dict_all = {}
        # for resd_name in RNA_CONSTANTS.RESD_NAMES:
        #     trans_dict = {}
        #     cord_dict = {}
        #
        #     atom_infos = RNA_CONSTANTS.ATOM_INFOS_PER_RESD[resd_name]
        #     angl_infos = RNA_CONSTANTS.ANGL_INFOS_PER_RESD[resd_name]
        #     n_angls = len(angl_infos)
        #
        #     for atom_name, idx_rgrp, _ in atom_infos:
        #         if idx_rgrp == 0:  # 主链坐标系下
        #             cord_dict[atom_name] = self.cord_dict[resd_name][atom_name]
        #
        #     trans_dict['omega-main'] = (torch.eye(3, dtype=torch.float32), torch.zeros((3), dtype=torch.float32))
        #     trans_dict['phi-main'] = (torch.eye(3, dtype=torch.float32), torch.zeros((3), dtype=torch.float32))
        #
        #     for idx_angl, (angl_name, _, atom_names_sel) in enumerate(angl_infos):
        #         x1 = cord_dict[atom_names_sel[0]]
        #         x2 = cord_dict[atom_names_sel[1]]
        #         x3 = cord_dict[atom_names_sel[2]]
        #         rot, tsl_vec = calc_rot_tsl(x1, x3, x3 + (x3 - x2))  # x3-x2,x1-x3
        #         trans_dict['%s-main' % angl_name] = (rot, tsl_vec)  # 每个侧链坐标系到主链坐标系的转换
        #
        #         for atom_name, idx_rgrp, _ in atom_infos:
        #             if idx_rgrp == idx_angl + 3:  # 在主链坐标系下的局部坐标,cord_dict只是用来推导局部坐标系之间的转化？
        #                 # 把局部坐标系下的坐标转换为主链坐标系下的（这里不用考虑扭转角的rotation吗？）
        #                 cord_dict[atom_name] = tsl_vec + torch.sum(
        #                     rot * self.cord_dict[resd_name][atom_name].view(1, 3), dim=1)
        #
        #     for idx_angl_src in range(1, n_angls - 1):  # 在不同侧链局部坐标系（3->2,2->1)之间转换
        #         idx_angl_dst = idx_angl_src + 1
        #         angl_name_src = angl_infos[idx_angl_src][0]
        #         angl_name_dst = angl_infos[idx_angl_dst][0]
        #         rot_src, tsl_vec_src = trans_dict['%s-main' % angl_name_src]
        #         rot_dst, tsl_vec_dst = trans_dict['%s-main' % angl_name_dst]
        #         rot = torch.matmul(rot_src.transpose(1, 0), rot_dst)
        #         tsl_vec = torch.matmul(rot_src.transpose(1, 0), tsl_vec_dst - tsl_vec_src)
        #         trans_dict['%s-%s' % (angl_name_dst, angl_name_src)] = (rot, tsl_vec)
        #
        #     trans_dict_all[resd_name] = trans_dict

        self.trans_dict_init = TRANS_DICT_PER_RESD

    def build_cords(self, seq, fram, angl, rtn_cmsk=False):

        # initialization
        n_resds = len(seq)
        device = angl.device

        angl = angl.squeeze(dim=0) / (torch.norm(angl.squeeze(dim=0), dim=2, keepdim=True) + self.eps)
        if fram.size()[-1]==7:
            rigid = Rigid.from_tensor_7(fram, normalize_quats=True)
            fram = rigid.to_tensor_4x4()
        rot = fram[:, :, :3, :3]
        tsl = fram[:, :, :3, 3:].permute(0, 1, 3, 2)

        fram = torch.cat([rot, tsl], dim=2)[:, :, :4, :3].permute(1, 0, 2, 3)
        fmsk = torch.ones((n_resds, 1), dtype=torch.int8, device=device)
        amsk = torch.ones((n_resds, RNA_CONSTANTS.N_ANGLS_PER_RESD_MAX), dtype=torch.int8, device=device)
        cord = torch.zeros((n_resds, RNA_CONSTANTS.ATOM_NUM_MAX, 3), dtype=torch.float32, device=device)
        cmsk = torch.zeros((n_resds, RNA_CONSTANTS.ATOM_NUM_MAX), dtype=torch.int8, device=device)
        fram_dict = defaultdict(lambda: torch.zeros((n_resds, 4, 3), dtype=torch.float32, device=device))
        atm_name = np.full((n_resds, RNA_CONSTANTS.ATOM_NUM_MAX), 'X', dtype=object)

        for resd_name in RNA_CONSTANTS.RESD_NAMES:
            idxs = [x for x in range(n_resds) if seq[x] == resd_name]
            if len(idxs) == 0:
                continue
            cord[idxs], cmsk[idxs], atm_name[idxs], _fram_dict = \
                self.__build_cord(resd_name, fram[idxs], fmsk[idxs], angl[idxs], amsk[idxs])
            for k in _fram_dict:
                fram_dict[k][idxs] = _fram_dict[k]

        return (cord, cmsk, atm_name, fram_dict) if rtn_cmsk else (cord)

    def __build_cord(self, resd_name, fram, fmsk, angl, amsk):
        """"""

        # initialization
        device = fram.device
        n_resds = fram.shape[0]
        atom_names_all = RNA_CONSTANTS.ATOM_NAMES_PER_RESD[resd_name]
        atom_names_pad = atom_names_all + ['X'] * (RNA_CONSTANTS.ATOM_NUM_MAX - len(atom_names_all))
        atom_infos_all = RNA_CONSTANTS.ATOM_INFOS_PER_RESD[resd_name]

        cord_dict = defaultdict(
            lambda: torch.zeros((n_resds, 3), dtype=torch.float32, device=device))
        cmsk_vec_dict = defaultdict(lambda: torch.zeros((n_resds), dtype=torch.int8, device=device))

        fram_null = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32, device=device)
        fram_dict = defaultdict(lambda: fram_null.unsqueeze(dim=0).repeat(n_resds, 1, 1))
        fmsk_vec_dict = defaultdict(lambda: torch.zeros((n_resds), dtype=torch.int8, device=device))

        trans_dict = {'main': (fram[:, 0, :3], fram[:, 0, 3])}

        rot_curr, tsl_curr = trans_dict['main']
        atom_names_sel = [x[0] for x in atom_infos_all if x[1] == 0]
        for atom_name_sel in atom_names_sel:
            cord_vec = self.cord_dict[resd_name][atom_name_sel].to(device)
            cord_dict[atom_name_sel] = \
                tsl_curr + torch.sum(rot_curr * cord_vec.view(1, 1, 3), dim=2)
            cmsk_vec_dict[atom_name_sel] = fmsk[:, 0]

        # determine 3D coordinates of atoms belonging to side-chain rigid-groups
        angl_infos_all = RNA_CONSTANTS.ANGL_INFOS_PER_RESD[resd_name]
        # rgrp_names_all = ['omega', 'phi'] + [x[0] for x in angl_infos_all]
        rgrp_names_all = [x[0] for x in angl_infos_all]

        for idx_rgrp, rgrp_name_curr in enumerate(rgrp_names_all):
            if rgrp_name_curr in ['omega', 'phi', 'angl_0', 'angl_1']:  # 上一级是主链
                rgrp_name_prev = 'main'
            else:
                rgrp_name_prev = 'angl_%d' % (int(rgrp_name_curr[-1]) - 1)  # 上一级是另一个侧链坐标系

            rot_prev, tsl_prev = trans_dict[rgrp_name_prev]
            rot_base, tsl_vec_base = \
                self.trans_dict_init[resd_name]['%s-%s' % (rgrp_name_curr, rgrp_name_prev)]
            rot_base = torch.from_numpy(rot_base).unsqueeze(dim=0).to(device).float()  # 到上一个坐标系的转换
            tsl_base = torch.from_numpy(tsl_vec_base).unsqueeze(dim=0).to(device).float()

            rot_addi, tsl_addi = calc_angl_rot_tsl(angl[:, idx_rgrp])
            rot_curr, tsl_curr = merge_rot_tsl(
                rot_prev, tsl_prev, rot_base, tsl_base, rot_addi, tsl_addi)
            trans_dict[rgrp_name_curr] = (rot_curr, tsl_curr)  #

            fram_dict[rgrp_name_curr] = \
                torch.cat([rot_curr, tsl_curr.unsqueeze(dim=1)], dim=1)
            fmsk_vec_dict[rgrp_name_curr] = fmsk[:, 0] * amsk[:, idx_rgrp]

            # atom_names_sel = [x[0] for x in atom_infos_all if x[1] == idx_rgrp + 1]
            atom_names_sel = [x[0] for x in atom_infos_all if x[1] == idx_rgrp + 3]
            for atom_name_sel in atom_names_sel:
                cord_vec = self.cord_dict[resd_name][atom_name_sel].to(device)

                cord_dict[atom_name_sel] = \
                    tsl_curr + torch.sum(rot_curr * cord_vec.view(1, 1, 3), dim=2)
                cmsk_vec_dict[atom_name_sel] = fmsk_vec_dict[rgrp_name_curr]

        cmsk = torch.stack([cmsk_vec_dict[x] for x in atom_names_pad][:RNA_CONSTANTS.ATOM_NUM_MAX], dim=1)
        cord = torch.stack([cord_dict[x] for x in atom_names_pad][:RNA_CONSTANTS.ATOM_NUM_MAX], dim=1)
        atm_name = np.stack(
            [np.full_like(cmsk_vec_dict[x].detach().cpu().numpy().astype(object), x[0], dtype=object) for x in
             atom_names_pad][
            :RNA_CONSTANTS.ATOM_NUM_MAX], axis=1)

        return cord, cmsk, atm_name, fram_dict

    @staticmethod
    def export_pdb_file(seq, atom_cords, path=None, atom_names=RNA_CONSTANTS.ATOM_NAMES_PER_RESD,
                        n_key_atoms=RNA_CONSTANTS.ATOM_NUM_MAX, atom_masks=None, lens=None, resd1to3=None,
                        confidence=None, chain_id=None, retain_all_res=True, logger=None):
        """Export a PDB file."""

        # configurations
        i_code = ' '
        # chain_id = '0' if chain_id is None else chain_id
        occupancy = 1.0
        cord_min = -999.0
        cord_max = 999.0
        seq_len = len(seq)
        if lens is None: lens = [seq_len]

        # take all the atom coordinates as valid, if not specified
        if atom_masks is None:
            atom_masks = np.ones(atom_cords.shape[:-1], dtype=np.int8)

        # determine the set of atom names (per residue)
        if atom_cords.ndim == 2:
            if atom_cords.shape[0] == seq_len * n_key_atoms:
                atom_cords = np.reshape(atom_cords, [seq_len, n_key_atoms, 3])
                atom_masks = np.reshape(atom_masks, [seq_len, n_key_atoms])
            else:
                raise ValueError('atom coordinates\' shape does not match the sequence length')

        elif atom_cords.ndim == 3:
            assert atom_cords.shape[0] == seq_len
            atom_cords = atom_cords
            atom_masks = atom_masks

        else:
            raise ValueError('atom coordinates must be a 2D or 3D np.ndarray')

        # reset invalid values in atom coordinates
        atom_cords = np.clip(atom_cords, cord_min, cord_max)
        # atom_cords[np.isnan(atom_cords)] = 0.0
        # atom_cords[np.isinf(atom_cords)] = 0.0

        lines = []
        n_atoms = 0
        for ich, Lch in enumerate(lens):
            start = 0 if ich == 0 else sum(lens[:ich])
            end = sum(lens[:ich + 1])
            for idx_resd in range(start, end):
                resd_name = seq[idx_resd]
                # for idx_resd, resd_name in enumerate(seq):
                if resd_name not in ['A', 'G', 'U', 'C']:
                    if retain_all_res:
                        atms = ["C4'", "C1'", "C2'", "C3'", "C5'", "O2'", "O3'", "O4'", "O5'", 'P', 'OP1', 'OP2']
                    else:
                        continue
                else:
                    atms = atom_names[resd_name]
                for idx_atom, atom_name in enumerate(atms):
                    if np.isnan(atom_cords[idx_resd, idx_atom, 0]) or np.isinf(atom_cords[idx_resd, idx_atom, 0]):
                        continue
                    temp_factor = 0.0 if confidence is None else \
                        float(100 * confidence.reshape([seq_len])[idx_resd - 1])

                    if atom_masks[idx_resd, idx_atom] == 0:
                        continue
                    n_atoms += 1
                    charge = atom_name[0]
                    ires = int(idx_resd - start)
                    chainID = ich if chain_id is None else chain_id if isinstance(chain_id, str) else chain_id[ich]
                    resName = resd_name if resd1to3 is None else resd1to3[resd_name]
                    resName = ' ' * (4 - len(resName)) + resName

                    line_str = ''.join([
                        'ATOM  ',
                        '%5d' % n_atoms,
                        '  ' + atom_name + ' ' * (3 - len(atom_name)),
                        '%s' % resName,
                        ' %s' % chainID[0],
                        ' ' * (4 - len(str(ires + 1))),
                        '%s' % str(ires + 1),
                        '%s   ' % i_code,
                        '%8.3f' % atom_cords[idx_resd, idx_atom, 0],
                        '%8.3f' % atom_cords[idx_resd, idx_atom, 1],
                        '%8.3f' % atom_cords[idx_resd, idx_atom, 2],
                        '%6.2f' % occupancy,
                        '%6.2f' % temp_factor,
                        ' ' * 10,
                        '%2s' % charge,
                        '%2s' % ' ',
                    ])
                    assert len(line_str) == 80, 'line length must be exactly 80 characters: ' + line_str
                    lines.append(line_str + '\n')
        if path is not None:
            # export the 3D structure to a PDB file
            os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
            with open(path, 'w') as o_file:
                o_file.write(''.join(lines))
        if logger is not None:
            logger.info(f'    Export PDB file to {path}')
        return lines
