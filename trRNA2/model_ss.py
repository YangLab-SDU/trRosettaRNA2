import torch.cuda
import torch.nn as nn

# pkg_dir = os.path.abspath(str(Path(__file__).parent.parent))
# sys.path.insert(0, pkg_dir)
from .RNAformer import RNAformer

from .utils import Symm


class InputEmbedder(nn.Module):
    def __init__(self, dim=48, in_dim=46, device='cpu'):
        super(InputEmbedder, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_dim, affine=True)
        self.elu1 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_dim, dim, 1)
        self.linear1 = nn.Sequential(
            self.bn1,
            self.elu1,
            self.conv1
        )
        self.token_emb = nn.Embedding(5, dim)
        self.device = device

    def forward(self, msa, msa_cutoff=500):
        f2d = self.get_f2d(msa[0])
        pair = self.linear1(f2d.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        m = self.token_emb(msa[:, :msa_cutoff, :].long())
        return {'pair': pair, 'msa': m}

    def get_f2d(self, msa, ss=None):
        nrow, ncol = msa.size()[-2:]
        if nrow == 1:
            msa = msa.view(nrow, ncol).repeat(2, 1)
            nrow = 2
        msa1hot = (torch.arange(5).to(self.device) == msa[..., None].long()).float()
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
        if ss is not None:
            f2d = torch.cat([f2d, ss.unsqueeze(-1).float()], dim=-1)
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

    def fast_dca(self, msa1hot, weights, penalty=4.5):
        nr, nc, ns = msa1hot.size()
        try:
            x = msa1hot.view(nr, nc * ns)
        except RuntimeError:
            x = msa1hot.contiguous().view(nr, nc * ns)
        num_points = weights.sum() - torch.sqrt(weights.mean())

        mean = torch.sum(x * weights[:, None], dim=0, keepdim=True) / num_points
        x = (x - mean) * torch.sqrt(weights[:, None])
        cov = torch.matmul(x.permute(1, 0), x) / num_points

        cov_reg = cov + torch.eye(nc * ns).to(self.device) * penalty / torch.sqrt(weights.sum())
        inv_cov = torch.inverse(cov_reg)

        x1 = inv_cov.view(nc, ns, nc, ns)
        x2 = x1.permute(0, 2, 1, 3)
        features = x2.reshape(nc, nc, ns * ns)

        x3 = torch.sqrt((x1[:, :-1, :, :-1] ** 2).sum((1, 3))) * (1 - torch.eye(nc).to(self.device))
        apc = x3.sum(dim=0, keepdim=True) * x3.sum(dim=1, keepdim=True) / x3.sum()
        contacts = (x3 - apc) * (1 - torch.eye(nc).to(self.device))

        return torch.cat([features, contacts[:, :, None]], dim=2)


class RecyclingEmbedder(nn.Module):
    def __init__(self, dim=48):
        super(RecyclingEmbedder, self).__init__()
        self.norm_pair = nn.LayerNorm(dim)
        self.norm_msa = nn.LayerNorm(dim)

    def forward(self, reprs_prev):
        pair = self.norm_pair(reprs_prev['pair'])
        single = self.norm_msa(reprs_prev['single'])
        return single, pair


class SSpredictor(nn.Module):
    def __init__(self, dim_2d=48, layers_2d=12, config={}, device='cpu'):
        super(SSpredictor, self).__init__()

        self.input_embedder = InputEmbedder(dim=dim_2d, device=device)
        self.recycle_embedder = RecyclingEmbedder(dim=dim_2d)
        self.net2d = RNAformer(dim=dim_2d,
                               depth=layers_2d,
                               msa_tie_row_attn=config['RNAformer']['msa_tie_row_attn'],
                               attn_dropout=config['RNAformer']['dropout_rate_attn'],
                               ff_dropout=config['RNAformer']['dropout_rate_ff'],
                               use_r2n=config['RNAformer']['use_r2n'],
                               qknorm=config['RNAformer']['qknorm'],
                               )
        self.to_ss = nn.Sequential(
            nn.LayerNorm(dim_2d),
            nn.Linear(dim_2d, dim_2d),
            Symm('b i j d->b j i d'),
            nn.Linear(dim_2d, dim_2d),
            nn.ReLU(),
            nn.LayerNorm(dim_2d),
            nn.Dropout(.1),
            nn.Linear(dim_2d, 1)
        )

    def forward(self, msa, res_id=None, num_recycle=3, msa_cutoff=500, training=False):
        reprs_prev = None
        for c in range(1 + num_recycle):
            with torch.set_grad_enabled(training and c == num_recycle):
                # with torch.amp.autocast(enabled=True,device_type='cuda'):
                reprs = self.input_embedder(msa if msa.ndim == 3 else msa[None], msa_cutoff=msa_cutoff)

                if reprs_prev is None:
                    reprs_prev = {
                        'pair': torch.zeros_like(reprs['pair']),
                        'single': torch.zeros_like(reprs['msa'][:, 0]),
                        'x': torch.zeros(list(reprs['pair'].shape[:2]) + [3], device=reprs['pair'].device),
                    }
                rec_msa, rec_pair = self.recycle_embedder(reprs_prev)
                reprs['msa'][:, 0] = reprs['msa'][:, 0] + rec_msa
                reprs['pair'] = reprs['pair'] + rec_pair
                out = self.net2d(reprs['pair'], msa_emb=reprs['msa'], return_msa=True, res_id=res_id,
                                 preprocess=False, return_attn=c == num_recycle)
                if c != num_recycle:
                    pair_repr, msa_repr = out
                else:
                    pair_repr, msa_repr, attn_maps = out
                reprs_prev = {
                    'single': msa_repr[..., 0, :, :].detach(),
                    'pair': pair_repr.detach(),
                    # "x": outputs["cords_c1'"][-1].detach(),
                }

        pred_ss = self.to_ss(pair_repr).sigmoid()

        return pred_ss
