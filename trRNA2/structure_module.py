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

import math
from typing import Optional, Tuple, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils_3d.converter import RNAConverter

from .utils_3d.primitives import Linear, LayerNorm
from .utils_3d.rigid_utils import Rigid

from .utils_3d.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)


class PosEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        if self.padding_idx is None:
            mask = torch.ones_like(input).int()
        else:
            mask = input.ne(self.padding_idx).int()

        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long()
        if self.padding_idx is not None:
            positions = positions + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden)
        self.linear_2 = Linear(self.c_hidden, self.c_hidden)

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
            self, s: torch.Tensor, s_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s


def weighted_softmax(x, weights, dim=-1, ss_usage='multiply'):
    max_x = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)

    if ss_usage in ['linear', 'multiply']:
        weighted_exp_x = exp_x * weights
    elif ss_usage == 'sum':
        weighted_exp_x = exp_x + weights
    else:
        raise ValueError(f'ss_usage {ss_usage} not supported, should be "multiply" or "sum"')

    softmax_x = weighted_exp_x / torch.sum(weighted_exp_x, dim=dim, keepdim=True)

    return softmax_x


class InvariantPointAttention(nn.Module):

    def __init__(
            self,
            c_s: int,
            c_z: int,
            c_hidden: int,
            no_heads: int,
            no_qk_points: int,
            no_v_points: int,
            inf: float = 1e5,
            eps: float = 1e-8,
            ss_usage='multiply',
    ):

        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        # hpv = self.no_heads * self.no_v_points * 3

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        # ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
                self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.ss_usage = ss_usage
        if ss_usage == 'linear':
            self.linear_ss = nn.Conv2d(1, no_heads, 1)

    def forward(
            self,
            s: torch.Tensor,
            z: Optional[torch.Tensor],
            r: Rigid,
            mask: torch.Tensor,
            ss: torch.Tensor = None,
            t_ss: float = 1,
            s_bias: torch.Tensor = None,
            a_bias: torch.Tensor = None,
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
            return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
            ss
                [*, N_res, N_res]
            t_ss: temperature for SS
        Returns:
            [*, N_res, C_s] single representation update
        """
        z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################

        if s_bias is not None:
            s = s + self.proj_bias1d(s_bias)

        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        if z[0] is not None:
            # [*, N_res, N_res, H]
            b = self.linear_b(z[0])

            if (_offload_inference):
                z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        if z[0] is not None:
            a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)

        pt_att = pt_att ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )

        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        if a_bias is not None:
            a = a + self.proj_bias2d(a_bias)

        if ss is None:
            a = self.softmax(a)
        else:
            if not hasattr(self, 'ss_usage'):
                self.ss_usage = 'multiply'
            batch_size, _, length, _ = a.size()
            ss = ss.view(batch_size, length, length)
            if t_ss is None or self.ss_usage == 'sum':
                weights = ss
            elif self.ss_usage == 'linear':
                weights = self.linear_ss(ss.view(batch_size, 1, length, length))
            else:
                weights = torch.exp(ss / t_ss) - 0.99
            a = weighted_softmax(a, weights, dim=-1, ss_usage=self.ss_usage)

        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if z[0] is not None:
            if (_offload_inference):
                z[0] = z[0].to(o_pt.device)

            # [*, N_res, H, C_z]
            o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

            # [*, N_res, H * C_z]
            o_pair = flatten_final_dims(o_pair, 2)

            # [*, N_res, C_s]
            s = self.linear_out(
                torch.cat(
                    (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
                ).to(dtype=z[0].dtype)
            )
        else:
            s = self.linear_out(
                torch.cat(
                    (o, *torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1
                ).to(dtype=s.dtype)
            )
        if return_attn:
            return s, a
        return s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector 
        """
        # [*, 6]
        update = self.linear(s)

        return update


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c)
        self.linear_2 = Linear(self.c, self.c)
        self.linear_3 = Linear(self.c, self.c)

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers):
        super(StructureModuleTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        for l in self.layers:
            s = l(s)

        s = self.layer_norm(s)

        return s


class StructureModule(nn.Module):
    def __init__(
            self,
            c_s,
            c_z,
            c_ipa,
            c_resnet,
            no_heads_ipa,
            no_qk_points,
            no_v_points,
            no_blocks,
            no_transition_layers,
            no_resnet_blocks,
            no_angles,
            trans_scale_factor,
            ss_usage='multiply',
            time_aware=False,
    ):
        super(StructureModule, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = 1e-8
        self.inf = 1e5

        self.default_frames = None
        self.group_idx = None
        self.atom_mask = None
        self.lit_positions = None
        self.time_aware = time_aware
        if time_aware:
            self.t_embedder = nn.Embedding(no_blocks, self.c_s)

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
            ss_usage=ss_usage,
        )

        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
        )

        self.bb_update = BackboneUpdate(self.c_s)

        self.angle_resnet = AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

        self.converter = RNAConverter()

        self.ss_usage = ss_usage

    def forward(
            self,
            seq,
            reprs,
            ss=None,
            use_t_ss=True,
            t_ss=[2 ** i for i in range(8)],
            mask=None,
            rigids=None,
            _offload_inference=False,
            _no_blocks=None,
            return_mid=False,
            allatm=True,
    ):
        s = reprs["single"]

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        z_reference_list = None
        z = None
        if reprs["pair"] is not None:
            # [*, N, N, C_z]
            z = self.layer_norm_z(reprs["pair"])
            if (_offload_inference):
                reprs["pair"] = reprs["pair"].cpu()
                z_reference_list = [z]
                z = None

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid.identity(
            s.shape[:-1],
            s.dtype,
            s.device,
            self.training,
            fmt="quat",
            # fmt="rot_mat",
        ) if rigids is None else rigids

        outputs = []
        grad_s = None
        grad_a = None
        n_blocks_act = self.no_blocks if _no_blocks is None else _no_blocks
        if not hasattr(self, 'time_aware'):
            self.time_aware = False
        for i in range(n_blocks_act):
            if self.time_aware:
                s = s + self.t_embedder(torch.Tensor([i]).long().to(s.device)).view(1, 1, -1)
            # [*, N, C_s]
            ds, a = self.ipa(
                s,
                z,
                rigids,
                mask,
                ss=ss,
                t_ss=t_ss[i] if use_t_ss else None,
                s_bias=grad_s,
                a_bias=grad_a,
                _offload_inference=_offload_inference,
                _z_reference_list=z_reference_list,
                return_attn=True
            )

            s = s + ds
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                'single': s
            }

            outputs.append(preds)

            if i != n_blocks_act - 1:
                rigids = rigids.stop_rot_gradient()

        del z, z_reference_list

        if (_offload_inference):
            reprs["pair"] = (
                reprs["pair"].to(s.device)
            )

        outputs = dict_multimap(torch.stack, outputs)
        outputs['unscaled_rigids'] = rigids

        if allatm:
            if return_mid:
                cord_list = []
                for frames, angles in zip(outputs['frames'], outputs['angles']):
                    cords, mask, atm_name, fram_dict = self.converter.build_cords(seq, frames, angles, rtn_cmsk=True)
                    cord_list.append([cords, mask, atm_name, fram_dict])
            else:
                cords, mask, atm_name, fram_dict = self.converter.build_cords(seq, outputs['frames'][-1],
                                                                              outputs['angles'][-1],
                                                                              rtn_cmsk=True)
                cord_list = [[cords, mask, atm_name, fram_dict]]
            if s.shape[1] > 100:
                torch.cuda.empty_cache()
            outputs['cord_tns_pred'] = [rearrange(
                rearrange(cord[0], '(b l) n d->b (l n) d', b=s.shape[0]),
                'b (l n) d->b n l d', l=s.shape[1]) for cord in cord_list]

            outputs['cmask'] = [cord[1].permute(1, 0)[None] for cord in cord_list]
            outputs['atm_name'] = [cord[2].T[None] for cord in cord_list]
            outputs["cords_c1'"] = [cord[0][:, 1, :].unsqueeze(0) for cord in cord_list]
            outputs["frames_allatm"] = {k: torch.stack([cord[3][k] for cord in cord_list], dim=0) for k in
                                        ['angl_0', 'angl_1', 'angl_2', 'angl_3']}

        return outputs
