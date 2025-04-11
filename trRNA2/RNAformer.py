import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
from torch.utils.checkpoint import checkpoint

from einops import rearrange
from einops.layers.torch import Rearrange

from functools import partialmethod
from typing import Union, List
import math


class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
        """
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x = x * mask
        return x


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class LayerNormNoBias(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True):
        super(LayerNormNoBias, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.gamma = torch.ones(normalized_shape)

    def forward(self, x):
        # 计算标准差
        std = torch.sqrt(x.var(dim=-1, unbiased=False, keepdim=True) + self.eps)
        # 规范化
        normalized_x = self.gamma * (x) / std
        return normalized_x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dilation=1, baseWidth=26, scale=4, stype='normal', expansion=4,
                 shortcut=True):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
        self.expansion = expansion

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.InstanceNorm2d(inplanes, affine=True)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation))
            bns.append(nn.InstanceNorm2d(width, affine=True))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.InstanceNorm2d(width * scale, affine=True)

        self.conv_st = nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1)

        self.relu = nn.ELU(inplace=True)
        self.stype = stype
        self.scale = scale
        self.width = width
        self.shortcut = shortcut

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.relu(self.bns[i](sp))
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        if self.stype == 'stage':
            residual = self.conv_st(residual)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.shortcut:
            out += residual

        return out


class TriangleMultiplication(nn.Module):
    def __init__(self, in_dim=128, dim=128, direct='outgoing'):
        super(TriangleMultiplication, self).__init__()
        self.direct = direct
        self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, dim * 2)
        self.linear2 = nn.Sequential(
            nn.Linear(in_dim, dim * 2),
            nn.Sigmoid()
        )
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.linear_out = nn.Linear(dim, in_dim)
        # self.linear_out.weight.data.fill_(0.)
        # self.linear_out.bias.data.fill_(0.)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            self.linear_out
        )

    def forward(self, z):
        direct = self.direct
        z = self.norm(z)
        a, b = torch.chunk(self.linear2(z) * self.linear1(z), 2, -1)
        gate = self.to_gate(z)
        if direct == 'outgoing':
            prod = torch.einsum('bikd,bjkd->bijd', a, b)
        elif direct == 'incoming':
            prod = torch.einsum('bkid,bkjd->bijd', a, b)
        else:
            raise ValueError('direct should be outgoing or incoming!')
        out = gate * self.to_out(prod)
        return out


class TriangleAttention(nn.Module):
    def __init__(self, in_dim=128, dim=32, n_heads=4, wise='row', qknorm=False):
        super(TriangleAttention, self).__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        if qknorm:
            self.q_norm = LayerNormNoBias(dim)
            self.k_norm = LayerNormNoBias(dim)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, n_heads * dim),
            nn.Sigmoid()
        )
        self.to_out = nn.Linear(n_heads * dim, in_dim)
        self.qknorm = qknorm
        # self.to_out.weight.data.fill_(0.)
        # self.to_out.bias.data.fill_(0.)

    def forward(self, z):
        wise = self.wise
        z = self.norm(z)
        q, k, v = torch.chunk(self.to_qkv(z), 3, -1)
        q, k, v = map(lambda x: rearrange(x, 'b i j (h d)->b i j h d', h=self.n_heads), (q, k, v))
        b = self.linear_for_pair(z)
        gate = self.to_gate(z)
        scale = q.size(-1) ** .5
        if wise == 'row':
            eq_attn = 'brihd,brjhd->brijh'
            eq_multi = 'brijh,brjhd->brihd'
            b = rearrange(b, 'b i j (r h)->b r i j h', r=1)
            softmax_dim = 3
        elif wise == 'col':
            eq_attn = 'bilhd,bjlhd->bijlh'
            eq_multi = 'bijlh,bjlhd->bilhd'
            b = rearrange(b, 'b i j (l h)->b i j l h', l=1)
            softmax_dim = 2

        else:
            raise ValueError('wise should be col or row!')
        if hasattr(self, 'qknorm') and self.qknorm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        attn = (torch.einsum(eq_attn, q, k / scale) + b).softmax(softmax_dim)
        out = torch.einsum(eq_multi, attn, v)
        out = gate * rearrange(out, 'b i j h d-> b i j (h d)')
        z_ = self.to_out(out)
        return z_


class PairTransition(nn.Module):
    def __init__(self, dim=128, n=4):
        super(PairTransition, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * n)
        self.linear2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim * n, dim)
        )

    def forward(self, z):
        z = self.norm(z)
        a = self.linear1(z)
        z = self.linear2(a)
        return z


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class PreNormCross(nn.Module):
    def __init__(self, dim1, dim2, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim1)
        self.norm_context = nn.LayerNorm(dim2)

    def forward(self, x, context, *args, **kwargs):
        x = self.norm(x)
        context = self.norm_context(context)
        return self.fn(x, context, *args, **kwargs)


class ToZero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.feed_forward(x)


class TriUpdate(nn.Module):
    def __init__(
            self,
            in_dim=128,
            n_heads=4,
            dim_pair_multi=64,
            dropout_rate_pair=.10,
            use_r2n=True, qknorm=False
    ):
        super(TriUpdate, self).__init__()

        self.ps_dropout_row_layer = DropoutRowwise(dropout_rate_pair)
        self.ps_dropout_col_layer = DropoutColumnwise(dropout_rate_pair)

        self.pair_multi_out = TriangleMultiplication(in_dim=in_dim, dim=dim_pair_multi, direct='outgoing')
        self.pair_multi_in = TriangleMultiplication(in_dim=in_dim, dim=dim_pair_multi, direct='incoming')

        dim_pair_attn = in_dim / n_heads
        assert dim_pair_attn == int(dim_pair_attn)
        dim_pair_attn = int(dim_pair_attn)

        self.pair_row_attn = TriangleAttention(in_dim=in_dim, dim=int(dim_pair_attn), n_heads=n_heads,
                                               qknorm=qknorm, wise='row')
        self.pair_col_attn = TriangleAttention(in_dim=in_dim, dim=int(dim_pair_attn), n_heads=n_heads,
                                               qknorm=qknorm, wise='col')

        self.pair_trans = PairTransition(dim=in_dim)

        self.conv_stem = nn.ModuleList(
            [
                nn.Sequential(
                    Rearrange('b i j d->b d i j'),
                    Bottle2neck(in_dim, in_dim, expansion=1, dilation=1, shortcut=False),
                    Rearrange('b d i j->b i j d'),
                ) if use_r2n else ToZero()
                for _ in range(4)
            ]
        )

    def forward(self, z, ckpt=True):
        z = z + self.ps_dropout_row_layer(self.pair_multi_out(z)) + self.conv_stem[0](z)
        z = z + self.ps_dropout_row_layer(self.pair_multi_in(z)) + self.conv_stem[1](z)
        pair_row_attn = self.pair_row_attn
        args = (z,)
        if z.requires_grad and ckpt:
            z = z + self.ps_dropout_row_layer(checkpoint(pair_row_attn, *args)) + self.conv_stem[2](z)
        else:
            z = z + self.ps_dropout_row_layer(pair_row_attn(*args)) + self.conv_stem[2](z)
        pair_col_attn = self.pair_col_attn
        args = (z,)
        if z.requires_grad and ckpt:
            z = z + self.ps_dropout_row_layer(checkpoint(pair_col_attn, *args)) + self.conv_stem[3](z)
        else:
            z = z + self.ps_dropout_row_layer(pair_col_attn(*args)) + self.conv_stem[3](z)
        z = z + self.pair_trans(z)

        return z


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_pair=None,
            conv_in_head=False,
            heads=8,
            dim_head=64,
            dropout=0.,
            tie_attn_dim=None
    ):
        super().__init__()

        self.scale = dim_head ** -0.5
        if conv_in_head:
            heads = 9
            self.to_q_kv = nn.ModuleList([
                nn.ModuleList([nn.Conv1d(dim, dim_head, kernel_size=k1, padding=int((k1 - 1) / 2), bias=False),
                               nn.Conv1d(dim, dim_head * 2, kernel_size=k2, padding=int((k2 - 1) / 2), bias=False)])
                for k1 in [1, 3, 5] for k2 in [1, 3, 5]
            ])
            self.conv_in_head = conv_in_head
            inner_dim = dim_head * heads
        else:
            inner_dim = dim_head * heads
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.heads = heads
        self.to_out = nn.Linear(inner_dim, dim)
        self.pair_norm = nn.LayerNorm(dim_pair)
        self.pair_linear = nn.Linear(dim_pair, heads, bias=False)

        self.for_pair = nn.Sequential(
            self.pair_norm, self.pair_linear
        )

        self.dropout = nn.Dropout(dropout)

        self.tie_attn_dim = tie_attn_dim
        self.seq_weight = PositionalWiseWeight(n_heads=heads, d_msa=dim)

    def forward(self, *args, context=None, tie_attn_dim=None, return_attn=False, soft_tied=False):
        if len(args) == 2:
            x, pair_bias = args
        elif len(args) == 1:
            x, pair_bias = args[0], None
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)
        # orig: (B*R, L, D)
        context = default(context, x)

        if hasattr(self, 'conv_in_head') and self.conv_in_head:
            x = rearrange(x, 'b n d->b d n')
            context = rearrange(context, 'b n d->b d n')
            qs, ks, vs = [], [], []
            for to_q, to_kv in self.to_q_kv:
                _q, _k, _v = (to_q(x), *to_kv(context).chunk(2, dim=1))
                qs.append(_q)
                ks.append(_k)
                vs.append(_v)
            q, k, v = map(lambda t: rearrange(torch.stack(t, dim=1), 'b h d n-> b h n d'), (qs, ks, vs))
        else:
            q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # for tying row-attention, for MSA axial self-attention

        if exists(tie_attn_dim):
            q, k, v = map(lambda t: rearrange(t, '(b r) h n d -> b r h n d', r=tie_attn_dim), (q, k, v))
            if soft_tied:
                w = self.seq_weight(rearrange(x, '(b r) l d -> b r l d', r=tie_attn_dim))  # b, L, H, R
                dots = einsum('b i h r, b r h i d, b r h j d -> b h i j', w, q, k) * self.scale
            else:
                dots = einsum('b r h i d, b r h j d -> b h i j', q, k) * self.scale * (tie_attn_dim ** -0.5)

        else:
            # q, k, v = map(lambda t: rearrange(t, '(b r) h n d -> b r h n d', r=tie_attn_dim), (q, k, v))

            #  SA:(B R H L D), (B R H L D) -> (B H R L L)
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # b=R

        # attention
        if pair_bias is not None:
            dots += rearrange(self.for_pair(pair_bias), 'b i j h -> b h i j')  # b=1
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate

        if exists(tie_attn_dim):
            out = einsum('b h i j, b r h j d -> b r h i d', attn, v)
            out = rearrange(out, 'b r h n d -> (b r) h n d')
        else:
            out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # combine heads and project out
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if return_attn:
            return rearrange(out, '(b r) n d -> b r n d', b=1), attn.mean(0)
        else:
            return rearrange(out, '(b r) n d -> b r n d', b=1)


class MSAAttention(nn.Module):
    def __init__(
            self,
            tie_row_attn=False,
            use_conv=None,
            attn_class=SelfAttention,
            dim=64,
            **kwargs
    ):
        super().__init__()

        self.tie_row_attn = tie_row_attn  # tie the row attention, from the paper 'MSA Transformer'

        self.use_conv = use_conv
        conv_in_head = False
        if use_conv == 'before':
            self.conv = nn.Conv1d(dim, dim, 3, padding=1)
        elif use_conv == 'head':
            conv_in_head = True
        self.attn_width = attn_class(dim, conv_in_head=conv_in_head, **kwargs)
        self.attn_height = attn_class(dim, **kwargs)

    def forward(self, *args, return_attn=False, ckpt=True):
        if len(args) == 2:
            x, pair_bias = args
        if len(args) == 1:
            x, pair_bias = args[0], None
        if len(x.shape) == 5:
            assert x.size(1) == 1, f'x has shape {x.size()}!'
            x = x[:, 0, ...]

        b, h, w, d = x.size()

        if hasattr(self, 'use_conv') and self.use_conv == 'before':
            x = rearrange(self.conv(rearrange(x, 'b h w d->b d (h w)')), 'b d (h w)->b h w d', h=h)

        # col-wise
        w_x = rearrange(x, 'b h w d -> (b w) h d')
        if w_x.requires_grad and ckpt:
            w_out = checkpoint(self.attn_width, w_x)
        else:
            w_out = self.attn_width(w_x)

        # row-wise
        tie_attn_dim = x.shape[1] if self.tie_row_attn else None
        h_x = rearrange(x, 'b h w d -> (b h) w d')
        attn_height = partial(self.attn_height, tie_attn_dim=tie_attn_dim, return_attn=return_attn)

        if h_x.requires_grad and ckpt:
            h_out = checkpoint(attn_height, h_x, pair_bias)
        else:
            h_out = attn_height(h_x, pair_bias)
        if return_attn:
            h_out, attn = h_out

        out = w_out.permute(0, 2, 1, 3) + h_out
        out /= 2
        if return_attn:
            return out, attn
        return out


class PositionalWiseWeight(nn.Module):
    def __init__(self, d_msa=128, n_heads=4):
        super(PositionalWiseWeight, self).__init__()
        self.to_q = nn.Linear(d_msa, d_msa)
        self.to_k = nn.Linear(d_msa, d_msa)
        self.n_heads = n_heads

    def forward(self, m):
        q = self.to_q(m[:, 0:1, :, :])  # b,1,L,d
        k = self.to_k(m)  # b,L,L,d

        q = rearrange(q, 'b i j (h d) -> b j h i d', h=self.n_heads)
        k = rearrange(k, 'b i j (h d) -> b j h i d', h=self.n_heads)
        scale = (q.size(-1) + 1e-8) ** .5
        attn = torch.einsum('bjhud,bjhid->bjhi', q, k) / scale
        return attn.softmax(dim=-1)  # b, L, H, R


class UpdateX(nn.Module):
    def __init__(self, in_dim=128, dim_msa=32, dim=128):
        super(UpdateX, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        # self.seq_weight = PositionalWiseWeight(in_dim)
        # self.proj_down2 = nn.Linear(dim_msa ** 2 * 4 + dim + 8, dim)
        self.proj_down2 = nn.Linear(dim_msa ** 2, dim)
        self.elu = nn.ELU(inplace=False)
        self.bn1 = nn.InstanceNorm2d(dim, affine=True)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(dim, affine=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, m, w=None):
        m = self.proj_down1(m)  # b,r,l,d
        nrows = m.shape[1]
        outer_product = torch.einsum('brid,brjc -> bijcd', m, m) / nrows
        outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')
        outer_product = self.proj_down2(outer_product)
        # pair_feats = torch.cat([x, outer_product], dim=-1)
        pair_feats = x + outer_product
        # pair_feats = rearrange(pair_feats,'b i j d -> b d i j')
        # out = self.bn1(pair_feats)
        # out = self.elu(out)
        # out = self.conv1(out)
        # out = self.bn2(out)
        # out = self.elu(out)
        # out = self.conv2(out)
        # return rearrange(pair_feats + out, 'b d i j -> b i j d')
        return pair_feats


class UpdateM(nn.Module):
    def __init__(self, in_dim=128, pair_dim=128, n_heads=8):
        super(UpdateM, self).__init__()
        self.norm1 = nn.LayerNorm(pair_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(pair_dim, n_heads)
        self.linear2 = nn.Linear(in_dim, in_dim // n_heads)
        self.ff = FeedForward(in_dim, dropout=.1)
        self.n_heads = n_heads

    def forward(self, x, m):
        pair_feats = (x + rearrange(x, 'b i j d->b j i d')) / 2
        pair_feats = self.norm1(pair_feats)
        attn = self.linear1(pair_feats).softmax(-2)  # b i j h
        values = self.norm2(m)
        values = self.linear2(values)  # b r l d
        attn_out = torch.einsum('bijh,brjd->brihd', attn, values)
        attn_out = rearrange(attn_out, 'b r l h d -> b r l (h d)')
        out = m + attn_out
        residue = self.norm3(out)
        return out + self.ff(residue)


class relpos(nn.Module):

    def __init__(self, dim=128):
        super(relpos, self).__init__()
        self.linear = nn.Linear(65, dim)

    def forward(self, res_id):
        device = res_id.device
        bin_values = torch.arange(-32, 33, device=device)
        d = res_id[:, :, None] - res_id[:, None, :]
        bdy = torch.tensor(32, device=device)
        d = torch.minimum(torch.maximum(-bdy, d), bdy)
        d_onehot = (d[..., None] == bin_values).float()
        assert d_onehot.sum(dim=-1).min() == 1
        p = self.linear(d_onehot)
        return p


class InputEmbedder(nn.Module):
    def __init__(self, dim):
        super(InputEmbedder, self).__init__()
        self.relpos = relpos(dim=dim)

    def forward(self, z, res_id):
        z = z + self.relpos(res_id)
        return z


# main class
class BasicBlock(nn.Module):
    def __init__(self, dim=64, heads=8, dim_head=32, msa_tie_row_attn=False, msa_conv=None, attn_dropout=.1,
                 ff_dropout=.1, use_r2n=True, qknorm=False):
        super().__init__()
        prenorm = partial(PreNorm, dim)

        self.PairMSA2MSA = prenorm(
            MSAAttention(dim=dim, dim_pair=dim, heads=heads, dim_head=dim_head, dropout=attn_dropout,
                         tie_row_attn=msa_tie_row_attn, use_conv=msa_conv,
                         ))
        self.MSA_FF = prenorm(FeedForward(dim=dim, dropout=ff_dropout))
        self.MSA2Pair = UpdateX(in_dim=dim, dim=dim)
        self.Pair2Pair = TriUpdate(in_dim=dim, dropout_rate_pair=attn_dropout, use_r2n=use_r2n, qknorm=qknorm)
        self.Pair2MSA = UpdateM(in_dim=dim, pair_dim=dim)

    def forward(self, msa, pair, return_attn=False, ckpt=True):

        if return_attn:
            m_out, attn_map = self.PairMSA2MSA(msa, pair, return_attn=True, ckpt=ckpt)
            attn_map = rearrange(attn_map, 'h i j -> i j h')
        else:
            m_out = self.PairMSA2MSA(msa, pair, return_attn=False, ckpt=ckpt)
        msa = msa + m_out
        msa = msa + self.MSA_FF(msa)
        pair = self.MSA2Pair(pair, msa)
        pair = self.Pair2Pair(pair, ckpt=ckpt)
        msa = self.Pair2MSA(pair, msa)
        _reprs = {'msa': msa, 'pair': pair}

        if return_attn:
            _reprs['attn_map'] = attn_map
        return _reprs


class RNAformer(nn.Module):
    def __init__(
            self,
            *,
            dim=32,
            in_dim=526,
            emb_dim=640,
            depth=32,
            heads=8,
            dim_head=64,
            num_tokens=5,
            attn_dropout=0.,
            ff_dropout=0.,
            msa_tie_row_attn=False,
            msa_conv=None,
            use_r2n=True,
            qknorm=False,
    ):
        super().__init__()

        self.bn1 = nn.InstanceNorm2d(in_dim, affine=True)
        self.elu1 = nn.ELU(inplace=False)
        self.conv1 = nn.Conv2d(in_dim, dim, 1)
        self.linear1 = nn.Sequential(
            self.bn1,
            self.elu1,
            self.conv1
        )
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.linear_emb = nn.Linear(emb_dim, dim)
        self.input_emb = InputEmbedder(dim)

        # main trunk modules

        self.net = nn.ModuleList([
            BasicBlock(dim=dim, heads=heads, dim_head=dim_head, msa_tie_row_attn=msa_tie_row_attn, msa_conv=msa_conv,
                       attn_dropout=attn_dropout, ff_dropout=ff_dropout, use_r2n=use_r2n,
                       qknorm=qknorm)
            for _ in range(depth)
        ])

    def forward(
            self,
            f2d,
            msa=None,
            res_id=None,
            msa_emb=None,
            preprocess=True,
            return_msa=True,
            return_attn=False,
            return_mid=False,
            relpos_enc=True,
            ckpt=True
    ):
        device = f2d.device
        if preprocess:
            # add dca
            # x = torch.cat([x, f2d], dim=-1).permute(0, 3, 1, 2)
            x = f2d.permute(0, 3, 1, 2)
            x = self.linear1(x).permute(0, 2, 3, 1)

            # embed multiple sequence alignment (msa)

            m = self.token_emb(msa.long())
            if exists(msa_emb):
                m += self.linear_emb(msa_emb)
        else:
            x, m = f2d, msa_emb

        if res_id is not None or relpos_enc:
            if res_id is None:
                res_id = torch.arange(x.size(1), device=device)
            res_id = res_id.view(1, x.size(1))
            x = self.input_emb(x, res_id)

        attn_maps = []
        mid_reprs = []

        for layer in self.net:
            outputs = layer(m, x, return_attn=return_attn, ckpt=ckpt)
            torch.cuda.empty_cache()
            m = outputs['msa']
            x = outputs['pair']
            if return_attn:
                attn_maps.append(outputs['attn_map'])
            if return_mid:
                mid_reprs.append(outputs)

        out = [x]
        if return_msa:
            out.append(m)
        if return_attn:
            out.append(attn_maps)
        if return_mid:
            out.append(mid_reprs)
        return tuple(out)
