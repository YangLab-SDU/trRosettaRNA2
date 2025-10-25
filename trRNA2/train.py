import os
import re
import sys, json
import warnings
from functools import partial
from pathlib import Path
import logging

import torch
import torch.nn as nn

import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from time import time

from tqdm import tqdm

from .dataset import *
from .utils import *
from .model_3d import Folding

from .utils_train import *
from .utils_3d.converter import calc_angls
from .utils_3d.rigid_utils import calc_rot_tsl
from .utils_3d.constants import ANGL_INFOS_PER_RESD, ATOM_NUM_MAX, ATOM_NAMES_PER_RESD

import argparse

seed_everything(0)

parser = argparse.ArgumentParser(
    description='trRosettaRNA2 Model Training Script',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

path_group = parser.add_argument_group('Input and Output Paths')
path_group.add_argument('-fas', '--train_fasta_dir', type=str, required=True,
                        help='Directory containing FASTA files for the training set.')
path_group.add_argument('-npz', '--train_npz_dir', type=str, required=True,
                        help='Directory containing NPZ feature files for the training set.')
path_group.add_argument('-out', '--output_dir', default='trRNA2_train', type=str,
                        help='Directory to save trained models and logs.')
path_group.add_argument('-mname', '--model_name', default='model_1', type=str,
                        help='Name for the trained model checkpoint file.')
path_group.add_argument('-cdhit', '--cdhit_pth', default='./cd-hit-v4.5.5-2011-03-31', type=str,
                        help='Path to the CD-HIT executable installation directory.')

data_group = parser.add_argument_group('Training Data Settings')
data_group.add_argument('-lst', '--train_lst_file', type=str, default=None,
                        help='Optional: Path to a list file for filtering training samples by date. \n'
                             'If not provided, all samples in `train_npz_dir` will be used. \n'
                             'Can be downloaded via https://yanglab.qd.sdu.edu.cn/trRosettaRNA/benchmark/train10699.lst')
data_group.add_argument('-date', '--date_cutoff', default=20240101, type=int,
                        help='Release date cutoff for training samples (format: YYYYMMDD). '
                             'Requires `-lst` to be specified.')
data_group.add_argument('-crop_size', '--crop_size', nargs=3, type=int, default=[256, 384, 384],
                        metavar=('S1', 'S2', 'S3'),
                        help='Sequence crop sizes for the three training stages.')

hp_group = parser.add_argument_group('Model and Training Hyperparameters')
hp_group.add_argument('-lr', '--init_lr', nargs=3, default=[2e-4, 5e-5, 5e-5], type=float,
                      help='Initial learning rate for the three training stages.')
hp_group.add_argument('-bt', '--batch_size', default=1, type=int,
                      help='Effective batch size, achieved through gradient accumulation.')
hp_group.add_argument('-nrec', '--num_recycle', default=3, type=int,
                      help='Maximum number of recycles within the model.')
hp_group.add_argument('-save_per_epoch', '--save_per_epoch', action='store_true',
                      help='If set, save a model checkpoint after each epoch.')

sys_group = parser.add_argument_group('System and Execution Settings')
sys_group.add_argument('-gpu', '--gpu', default=0, type=int,
                       help='Specifies the GPU ID to use for training.')
sys_group.add_argument('-cpu', '--cpu', default=4, type=int,
                       help='Number of CPU threads for data loading.')
sys_group.add_argument('-debug', '--debug', action='store_true',
                       help='Run in debug mode with minimal data for quick checks.')
sys_group.add_argument('-warning', '--warning', action='store_true',
                       help='Display additional warning information during execution.')

args = parser.parse_args()

max_epochs = 200
loss_weights_stage1 = {
    'fape': 1, 'mid_fape': 1, 'torsion': 1, 'rev_crmsd': .5, 'ss2d': .2, 'geoms': .5, 'plddt': .05,
    'ss3d_loss1': .05, 'ss3d_loss2': 0.05, 'clash': .0, 'bond': .0,
}
loss_weights_stage23 = {
    'fape': 1, 'mid_fape': 1, 'torsion': 1, 'rev_crmsd': .5, 'ss2d': .2, 'geoms': .5, 'plddt': .1,
    'ss3d_loss1': .5, 'ss3d_loss2': 0.1, 'clash': 1, 'bond': .3,
}

model_name = args.model_name

config = {
    'lr': args.init_lr,
    'grad_accum': args.batch_size,
    'grad_clip': True,
    'max_recycle': args.num_recycle,
    'crop': {'train': args.crop_size, 'val': 800},
    'dim_pair': 64,
    'msa_cutoff': {'train': 200, 'test': 300},
    'use_ss': True,
    'init_str': 'nn',
    'ss3D': True,
    'divide': True,
    'loss_structure': {
        'd0_fape': 20,
        'weights': {
            1: loss_weights_stage1,
            2: loss_weights_stage23,
            3: loss_weights_stage23,
        }
    },
    'RNAformer': {
        'n_block': 12,
        'dropout_rate_attn': .3,
        'dropout_rate_ff': .3,
        'msa_tie_row_attn': False,
    },
    'structure_module': {
        "c_s": 64,
        "c_z": 64,
        "c_ipa": 16,
        "c_resnet": 128,
        "no_heads_ipa": 12,
        "no_qk_points": 4,
        "no_v_points": 8,
        "no_blocks": 8,
        "no_transition_layers": 1,
        "no_resnet_blocks": 2,
        "no_angles": 4,
        "trans_scale_factor": 10,
    }
}

early_stopping = True
patience_acc = 10
if not early_stopping:
    patience_acc = np.inf

torch.set_num_threads(2)
save_log = model_name

output_dir = args.output_dir


def calc_loss(outputs, native_feats, res_id, loss_weights):
    losses = {}
    loss_config = config['loss_structure']

    ############## 2D LOSS ##############
    # geometries loss
    geom_loss = 0
    for k in obj['inter_labels']:
        for a in obj['inter_labels'][k]:
            if k == 'contact':
                geom_loss += (cross_entropy_binary(
                    outputs['geoms']['inter_labels'][k][a].float(),
                    native_feats['geoms']['inter_labels'][k][a].to(device).float()
                ) / len(obj['inter_labels'][k]))
            else:
                geom_loss += (cross_entropy(
                    outputs['geoms']['inter_labels'][k][a].float(),
                    native_feats['geoms']['inter_labels'][k][a].to(device).float()
                ) / len(obj['inter_labels'][k]))
    losses['geoms'] = geom_loss

    # 2D SS loss
    losses['ss2d'] = cross_entropy_binary(outputs['ss'], native_feats['ss'])
    ############## 3D LOSS ##############
    # prepare
    p = np.array([0.8, 0.2])
    d_clamp = np.random.choice([loss_config['d0_fape'], np.inf], p=p.ravel())

    _FAPEloss = partial(FAPEloss, eps=1e-4, Z=loss_config['d0_fape'], d_clamp=d_clamp)

    native_angls = torch.stack([native_feats['angls'][f'angl_{angl_idx}'] for angl_idx in range(4)], dim=1)[None]
    native_cords_allatm = torch.clone(native_feats['cords_allatm'])
    native_frames = native_feats['frames']

    mainFrame_true = (native_frames['main'][0].view(1, 1, -1, 3, 3), native_frames['main'][1].view(1, 1, -1, 3))

    _mainFAPEloss = partial(_FAPEloss, x_true=mainFrame_true[1], frame_true=mainFrame_true)
    _allatmFAPEloss = partial(_FAPEloss, x_true=native_cords_allatm, frame_true=mainFrame_true,
                              cmask=outputs['cords_allatm_mask'])

    # auxiliary losses
    mid_loss = _mainFAPEloss(x=outputs['frames'][1], frame=outputs['frames']).mean()
    torsion_loss = torsionAngleLoss(outputs['unnormalized_angles'], native_angls)
    losses['mid_fape'] = mid_loss
    losses['torsion'] = torsion_loss

    # FAPE loss
    losses['fape'] = _allatmFAPEloss(x=outputs['cords_allatm'],
                                     frame=[arr[-1:] for arr in outputs['frames']]).mean()

    # RMSD
    crmsd = cRMSD(outputs['frames'][1][-1].squeeze(), native_frames['main'][1].squeeze())

    # LDDT
    _lddt = calculate_lddt(outputs['frames'][1][-1].squeeze(), native_frames['main'][1].squeeze(), rad=30)
    _lddt1hot = one_hot(_lddt, bin_values=torch.arange(.02, 1.02, .02, device=device))
    losses['plddt'] = cross_entropy(outputs['plddt_prob'][-1], _lddt1hot)

    # Clash loss and 3D SS loss
    clash_loss, ss3d_loss1, ss3d_loss2 = clash_ss3d_loss(outputs, native_feats)
    losses['clash'] = clash_loss
    losses['ss3d_loss1'] = ss3d_loss1
    losses['ss3d_loss2'] = ss3d_loss2

    # bond loss
    losses['bond'] = bond_loss(outputs['cords_allatm'], native_feats['seq'], res_id)

    total_loss = 0
    for k in losses:
        new_weight = loss_weights[k]
        if new_weight > 0 and losses[k] != 0 and not torch.isnan(losses[k]):
            total_loss += losses[k] * new_weight

    metrics = {'rmsd': crmsd, 'lddt': torch.nanmean(_lddt)}
    return total_loss, losses, metrics


def train(model, dataloader, stage=1, training=True):
    n_ERR = 0
    all_losses = defaultdict(list)
    all_metrics = defaultdict(list)
    val_table = {'loss': pd.DataFrame(), 'corr': pd.DataFrame(), 'precision': pd.DataFrame(), '3D': pd.DataFrame(), }

    min_ERR_shape = (np.inf, np.inf)
    if training:
        model.train()
    else:
        model.eval()
    idx_grad = 0
    for i, sample in tqdm(enumerate(dataloader)):
        if len(sample) == 0:
            n_ERR += 1
            continue

        msa = sample['msa'].to(device).long()
        ss = sample['ss'].to(device).float()
        res_id = sample['idx'].to(device).long()
        try:
            raw_seq = unparse_a3m(msa[0, 0:1])[0].replace('-', 'N')
        except Exception as e:
            print(msa.shape)
            raise e

        pid = sample['pid'][0]

        native_coor = {atom: sample['coords'][atom].float().to(device).squeeze(0) for atom in sample['coords']}
        if torch.isnan(native_coor['N1']).all() and torch.isnan(native_coor['N9']).all():
            n_ERR += 1
            continue

        native_feats = {}
        native_feats['geoms'] = {'inter_labels': sample['inter_labels']}
        native_feats['cords'] = native_coor
        native_feats['seq'] = raw_seq
        native_feats['seq_arr'] = msa[0, 0]
        try:
            native_feats['angls'] = calc_angls(native_coor, raw_seq)
        except Exception as err:
            print(f'{pid}, calc_angls err!: {str(err)}')
            n_ERR += 1
            continue

        b = msa.size(0)
        native_frames = {
            k: (np.nan * torch.zeros((b, len(raw_seq), 3, 3), device=device),
                np.nan * torch.zeros((b, len(raw_seq), 3), device=device))
            for k in ['main', 'angl_0', 'angl_1', 'angl_2', 'angl_3']}
        native_coor_allatm = np.nan * torch.zeros((b, ATOM_NUM_MAX, len(raw_seq), 3), device=device)
        for res in ANGL_INFOS_PER_RESD:
            indices = [i for i, aa in enumerate(raw_seq) if aa == res]
            if len(indices) == 0: continue
            for idx, atm in enumerate(ATOM_NAMES_PER_RESD[res]):
                if atm not in native_coor:
                    continue
                native_coor_allatm[:, idx, indices] = native_coor[atm][indices][None]
            atm1, atm2, atm3 = ["C4'", "N9", "C1'"] if res in ['A', 'G'] else ["C4'", "N1", "C1'"]
            _native_rot, _native_tsl = calc_rot_tsl(native_coor[atm1][indices], native_coor[atm3][indices],
                                                    native_coor[atm2][indices])
            native_frames['main'][0][0, indices] = _native_rot
            native_frames['main'][1][0, indices] = _native_tsl
            for idx_angl in range(4):
                atms = ANGL_INFOS_PER_RESD[res][idx_angl][-1]
                atm1, atm2, atm3 = atms[-1], atms[1], atms[2]
                _native_rot, _native_tsl = calc_rot_tsl(native_coor[atm1][indices], native_coor[atm3][indices],
                                                        native_coor[atm2][indices])
                native_frames[f'angl_{idx_angl}'][0][0, indices] = _native_rot
                native_frames[f'angl_{idx_angl}'][1][0, indices] = _native_tsl

        native_feats['cords_allatm'] = native_coor_allatm
        native_feats['frames'] = native_frames
        native_feats['ss'] = torch.where(torch.isnan(sample['native_ss']), 0, sample['native_ss']).to(device)

        num_recycle = random.choice(range(config['max_recycle'] + 1)) if training else 3
        try:
            with torch.set_grad_enabled(training):

                _, outputs = model(raw_seq=raw_seq, msa=msa, ss=ss, res_id=res_id, num_recycle=num_recycle,
                                   is_training=training, config=config)

                total_loss, losses, metrics = calc_loss(outputs, native_feats, res_id=res_id,
                                                        loss_weights=config['loss_structure']['weights'][stage])

                if not isinstance(total_loss, torch.Tensor) or torch.isnan(total_loss):
                    if args.warning:
                        logging.warning(f'{pid}\'s loss is nan! Skip this sample!')
                    del outputs, total_loss, losses, metrics
                    torch.cuda.empty_cache()
                    n_ERR += 1
                    continue

                if training:
                    all_parameters = []
                    for pg in optimizer.param_groups:
                        all_parameters += pg['params']
                    total_loss.backward(retain_graph=False)
                    nan_grad = False
                    if (idx_grad + 1) % config['grad_accum'] == 0:
                        if config['grad_clip']:
                            try:
                                torch.nn.utils.clip_grad_norm_(all_parameters, .5, error_if_nonfinite=True)
                            except RuntimeError as e:
                                if 'non-finite' in str(e):
                                    nan_grad = True
                        else:
                            for params in all_parameters:
                                if params.grad is not None and torch.isnan(params.grad).any():
                                    nan_grad = True
                                    break
                        if nan_grad:
                            if args.warning:
                                logging.warning(f'{pid}\'s gradients contain nan! Skip this sample!')
                            del outputs, total_loss, losses, metrics
                            torch.cuda.empty_cache()
                            n_ERR += (idx_grad % config['grad_accum'] + 1)
                            optimizer.zero_grad()
                            optimizer.step()
                            continue
                        else:
                            optimizer.step()
                            optimizer.zero_grad()
                    idx_grad += 1

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                torch.cuda.empty_cache()
                # if args.warning:
                logging.warning(f'{pid}\'s training is out of memory! Skip this sample!')
            else:
                raise exception
            n_ERR += 1
            if msa.size(-1) * msa.size(-2) < min_ERR_shape[0] * min_ERR_shape[1]:
                min_ERR_shape = msa.size()[-2:]
            continue

        # process losses and metrics
        all_losses['total'] = total_loss.detach().cpu().numpy()
        for k in losses:
            if isinstance(losses[k], torch.Tensor):
                all_losses[k].append(losses[k].detach().cpu().numpy())
                val_table['loss'].loc[pid, k] = losses[k].detach().cpu().numpy()

        for k in metrics:
            all_metrics[k].append(metrics[k].detach().cpu().numpy())
            val_table['3D'].loc[pid, k] = metrics[k].detach().cpu().numpy()

        del total_loss
    return all_losses, all_metrics, val_table, n_ERR


if __name__ == '__main__':

    os.makedirs(f'{output_dir}/models', exist_ok=True)
    os.makedirs(f'{output_dir}/config', exist_ok=True)
    os.makedirs(f'{output_dir}/tables', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{output_dir}/training.log"),
            logging.StreamHandler()
        ]
    )

    if args.train_lst_file is None:
        args.date_cutoff = None
    args_log = "\n--- Parsed Arguments ---\n"
    for key, value in vars(args).items():
        args_log += (f"{key:<20}: {value}\n")
    logging.info(args_log)

    torch.set_num_threads(args.cpu)

    if args.train_lst_file is None:
        all_lst = [f[:-4] for f in os.listdir(args.train_npz_dir) if f.endswith('.npz')]
        logging.info(f"All {len(all_lst)} samples will be used regardless of the release date")
    else:
        all_lst = []
        for line in open(args.train_lst_file).read().splitlines():
            pdbid, mid, chains, date = re.search(
                r'([0-9A-Z]{4}) model:([0-9]+) chains:([0-9A-Za-z,]+) release date: ([0-9]{8})', line).groups()
            if int(date) < args.date_cutoff:
                all_lst.append(f'{pdbid}_{mid}_{chains.replace(",", "_")}')
        logging.info(f"{len(all_lst)} samples released before {args.date_cutoff} will be used.")

    # perform sequence clustering
    with open(f'{output_dir}/all.fasta', 'w') as f:
        for pid in all_lst:
            f.write(open(f'{args.train_fasta_dir}/{pid}.fasta').read().rstrip() + '\n')
    train_lst, val_lst, train_clstrs = train_val_split(all_lst, args)
    logging.info(
        f"{len(train_lst)} for training (number of clusters: {len(train_clstrs)}), {len(val_lst)} for validation")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and int(args.gpu) > -1 else 'cpu')
    logging.info(f"Training on {device}.")

    structure_module_config = config['structure_module']
    save_to_json(config, f'{output_dir}/config/{model_name}.json')
    logging.info(f"Training configurations saved to {output_dir}/config/{model_name}.json.")

    # define dataset and dataloader
    train_dataset_stages = {
        1: trRNA2Dataset(train_clstrs, root_dir=args.train_npz_dir,
                         clusters=True, return_pid_lst=False,
                         random_=True, skip_loops=False,
                         lengthmax=config['crop']['train'][0], warning=args.warning),
        2: trRNA2Dataset(train_lst[:], root_dir=args.train_npz_dir,
                         clusters=False, return_pid_lst=True,
                         random_=True,
                         lengthmax=config['crop']['train'][1], warning=args.warning),
        3: trRNA2Dataset(train_lst[:], root_dir=args.train_npz_dir,
                         clusters=False, return_pid_lst=True,
                         random_=True, skip_loops=True,
                         lengthmax=config['crop']['train'][2], warning=args.warning),
    }
    val_dataset = trRNA2Dataset(val_lst[:], root_dir=args.train_npz_dir,
                                clusters=False, return_pid_lst=False,
                                random_=False, skip_loops=True,
                                lengthmax=config['crop']['val'], warning=args.warning)

    train_dataloader_stages = {k: DataLoader(train_dataset_stages[k], batch_size=1, shuffle=True) for k in
                               train_dataset_stages}
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = Folding(dim_2d=config['dim_pair'], layers_2d=config['RNAformer']['n_block'], config=config).to(device)

    # if args.pretrain_pth is not None:
    #     model_CKPT = torch.load(args.pretrain_pth, map_location=device)
    #     model.load_state_dict(model_CKPT['state_dict'])
    #     model.train()

    min_loss = np.inf
    max_lddt = 0

    steps_without_enhancing_loss = 0
    steps_without_enhancing_acc = 0
    s_ = time()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'][0])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=10, cooldown=0,
                                                           min_lr=1e-6)
    stage = 1

    logging.info("Starting training process...")
    logging.info(f"Epochs: {max_epochs}, Learning rate: {optimizer.param_groups[0]['lr']}")
    logging.info("Stage 1 starts")
    for epoch in range(max_epochs):
        logging.info(f"""\n-----------------epoch{epoch} lr:{optimizer.param_groups[0]['lr']}---------------------\n""")

        st = time()
        train_n_ERR = 0
        train_dataset = train_dataset_stages[stage]
        if stage == 3:
            if epoch == resample_init_epoch:
                logging.info("Assigning initial sample weights...")
                stage3_init_dataset = trRNA2Dataset(train_lst, root_dir=args.train_npz_dir,
                                                    clusters=False, return_pid_lst=True,
                                                    random_=False,
                                                    lengthmax=config['crop']['val'], warning=args.warning)
                tmp_dataloader = DataLoader(stage3_init_dataset, batch_size=1, shuffle=False, sampler=None)
                _, _, train_table, _ = train(model, tmp_dataloader, training=False)
                pid_lst = stage3_init_dataset.pid_lst
                train_lddt_dict = {}
            else:
                pid_lst = train_dataset.pid_lst

            sampler_weights = []
            for pid in pid_lst:
                try:
                    sampler_weights.append(max(0.05, 1 - train_table['3D'].loc[pid, 'lddt'] ** 2))
                    train_lddt_dict[pid] = train_table['3D'].loc[pid, 'lddt']
                except KeyError:
                    if pid in train_lddt_dict:
                        sampler_weights.append(max(0.05, 1 - train_lddt_dict[pid] ** 2))
                    else:
                        sampler_weights.append(0.05)
            np.save(f'{output_dir}/models/sampler_weight_{epoch}.npy', np.array(sampler_weights))
            num_samples = min(len(train_dataset), 5000)
            sampler = torch.utils.data.WeightedRandomSampler(sampler_weights, num_samples=num_samples, replacement=True)
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=sampler is None, sampler=sampler)

            sampler_weights = []
            for pid in pid_lst:
                try:
                    sampler_weights.append(max(0.05, 1 - train_table['3D'].loc[pid, 'lddt'] ** 2))
                except KeyError:
                    sampler_weights.append(0.2)
            np.save(f'{output_dir}/models/sampler_weight_{epoch}.npy', np.array(sampler_weights))
            num_samples = min(len(train_dataset), 10000)
            sampler = torch.utils.data.WeightedRandomSampler(sampler_weights, num_samples=num_samples, replacement=True)
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=sampler is None, sampler=sampler)
        else:
            train_dataloader = train_dataloader_stages[stage]

        # training
        if args.debug:
            with torch.autograd.detect_anomaly():
                train_losses, train_metrics, train_table, train_n_ERR = train(model, train_dataloader, stage=stage,
                                                                              training=True)
        else:
            train_losses, train_metrics, train_table, train_n_ERR = train(model, train_dataloader, stage=stage,
                                                                          training=True)

        save_dict = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(),
                     }

        if args.save_per_epoch:
            torch.save(save_dict, f'{output_dir}/models/{model_name}_stage{stage}_epoch{epoch}.pth.tar')
            logging.info(
                f'model state dict saved to {output_dir}/models/{model_name}_stage{stage}_epoch{epoch}.pth.tar')

        # validation
        model.eval()
        logging.info("Validating...")

        val_losses, val_metrics, val_table, val_n_ERR = train(model, val_dataloader, training=False)
        logging.info("Validation done")

        # save table
        writer = pd.ExcelWriter(f'{output_dir}/tables/{model_name}_stage{stage}_epoch{epoch}.xlsx', engine='openpyxl')
        for k in val_table:
            if k in train_table:
                train_table[k].sort_index(inplace=True)
                for col in train_table[k]:
                    train_table[k].loc['mean', col] = train_table[k][col].mean()
                train_table[k].to_excel(writer, sheet_name='train_' + k)
            val_table[k].sort_index(inplace=True)

            for col in val_table[k]:
                val_table[k].loc['mean', col] = val_table[k][col].mean()
            val_table[k].to_excel(writer, sheet_name='val_' + k)
        writer.close()
        logging.info(
            f"Detailed evaluation results saved to {output_dir}/tables/{model_name}_stage{stage}_epoch{epoch}.xlsx")

        # back to train mode
        model.train()
        e = time()

        logging.info(f"""time:{int(e - st)} seconds, skipped samples:tr {train_n_ERR} val {val_n_ERR} 
loss:train {np.nanmean(train_losses["total"]):.3f} val {np.nanmean(val_losses["total"]):.3f}
rmsd:train {np.nanmean(train_metrics["rmsd"]):.3f} val {np.nanmean(val_metrics["rmsd"]):.3f}
lddt:train {np.nanmean(train_metrics["lddt"]):.3f} val {np.nanmean(val_metrics["lddt"]):.3f}
""")

        metric = np.nanmean(val_metrics["lddt"])
        scheduler.step(metric)

        # early stopping
        if early_stopping:
            if metric <= max_lddt:
                steps_without_enhancing_acc += 1
            else:
                steps_without_enhancing_acc = 0
                max_lddt = max(metric, max_lddt)
                save_model(model.state_dict(), f'{output_dir}/models/{model_name}.pth.tar')
                logging.info(f'model state dict saved to {output_dir}/models/{model_name}.pth.tar')

            if steps_without_enhancing_acc >= patience_acc:
                if epoch < 80:
                    continue
                logging.info(f'epoch {epoch}: early stopping happened! ')
                if stage < 3:
                    steps_without_enhancing_acc = -5
                    stage += 1
                    logging.info(f'Now switch to stage {stage}')
                    for pg in optimizer.param_groups:
                        pg['lr'] = config['lr'][stage - 1]
                    if stage == 3:
                        resample_init_epoch = epoch + 1

                else:
                    logging.info(f'All training stages finished! ')
                    break
