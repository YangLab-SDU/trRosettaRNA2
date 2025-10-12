import logging
from collections import defaultdict
from pathlib import Path
from typing import List

import copy
import os.path
import zipfile

import zlib

import random

import warnings

import torch
import torch.nn as nn
import torch.distributed as distri
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset, DataLoader
import numpy as np
from os.path import join, isfile, isdir
from os import listdir

from .utils import subsample, parse_a3m, ss2mat, cosangle

bins_size = 1
dist_max = 40
dist_min = 3

obj = {
    'inter_labels': {
        'distance': ["C3'", "C4'", "P", "N1", "C4", "C1'", 'CiNj', 'PiNj'],
        'contact': ['all atom']
    }
}

n_bins = {
    'inter_labels': {
        'distance': int((dist_max - dist_min) / bins_size + 1),
        'contact': 1
    }
}
bins = {
    'distance': np.linspace(dist_min, dist_max, int((dist_max - dist_min) / bins_size + 1)),
}

warnings.filterwarnings("ignore", message="All-NaN axis encountered")


class trRNA2Dataset(Dataset):
    def __init__(self,
                 targets,
                 root_dir,
                 ss_dir=None,
                 ss_name='spot_pp,dssr_bps',
                 rowmax=20000,
                 lengthmax=200,
                 ntemp=20,
                 random_=True,
                 subsample_msa=True,
                 clusters=False,
                 return_coord=True,
                 return_single=False,
                 return_ss=True,
                 return_nativess=True,
                 cont_for_missed_SS=True,
                 nativess_type='bps',
                 nomsa_ok=False,
                 padding=False,
                 return_pid_lst=False,
                 skip_loops=False,
                 warning=False,
                 device=torch.device('cpu'),
                 ):
        super(trRNA2Dataset, self).__init__()

        self.targets = targets
        self.datadir = root_dir
        self.ss_name = ss_name
        self.rowmax = rowmax
        self.lengthmax = lengthmax
        self.ntemp = ntemp
        self.ss_dir = ss_dir
        self.return_coord = return_coord
        self.return_single = return_single
        self.return_ss = return_ss
        self.return_nativess = return_nativess
        self.cont_for_missed_SS = cont_for_missed_SS
        self.padding = padding
        self.random_ = random_
        self.subsample_msa = subsample_msa and random_
        self.skip_loops = skip_loops
        self.warning = warning
        self.device = device
        self.nomsa_ok = nomsa_ok

        self.nativess_type = nativess_type

        files = []
        if clusters:
            for clstr in self.targets:
                files_clstr = []
                for p in clstr:
                    sample_file = join(self.datadir, p + '.npz')

                    if not isfile(sample_file):
                        if self.warning:
                            warn = f'{sample_file} missed! Skip this sample!'
                            logging.warning(warn)
                        continue
                    files_clstr.append((sample_file, p[:-4]))
                files.append(files_clstr)
        else:
            for p in self.targets:
                if not p.endswith('.npz'): p = p + '.npz'
                sample_file = join(self.datadir, p)

                if not isfile(sample_file):
                    if self.warning:
                        warn = f'{sample_file} missed! Skip this sample!'
                        logging.warning(warn)
                    continue
                files.append((sample_file, p[:-4]))

                #  files list
        self.files = files
        if return_pid_lst:
            assert not clusters
            self.pid_lst = [l[-1] for l in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx, return_npz=False):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # file name
        file = self.files[idx]
        if isinstance(file, list):
            if file:
                file, pid = random.choice(file)
            else:
                if self.warning:
                    warn = f'no vaild npz file for cluster {idx}! Skip this sample!'
                    logging.warning(warn)
                return {}
        else:
            file, pid = file

        try:
            npz = np.load(file, allow_pickle=True)
            if self.return_coord:
                try:
                    coord_npz = npz['coords'].item()
                except KeyError:
                    print(pid, 'miss coord!')
                    return {}
            else:
                coord_npz = None

            seq = npz['seq']
            if not self.return_single:
                alns = {'seq': npz['seq']}
                if 'aln_noss' in npz.files:
                    alns = {'noss': npz['aln_noss']}
                    if 'aln_ss' in npz.files:
                        alns['ss'] = npz['aln_ss']
                    else:
                        alns['ss'] = alns['noss']
                    if self.random_:
                        aln_type = random.choice(list(alns.keys()))
                    else:
                        aln_type = 'ss'
                elif 'aln' in npz.files:
                    alns = {'aln': npz['aln']}
                    aln_type = 'aln'
                else:
                    aln_type = 'seq'
                msa = alns[aln_type]
            else:
                msa = npz['seq'].reshape(1, -1)


        except (zipfile.BadZipFile, zlib.error):
            if self.warning:
                warn = f'npz file {file} is broken! Skip this sample!'
                logging.warning(warn)
            return {}
        if self.return_nativess:
            if f'dssr_{self.nativess_type}' in npz:
                native_ss = npz[f'dssr_{self.nativess_type}']
                if len(native_ss.shape) == 3:
                    native_ss = native_ss[0]
                bps = np.nansum(native_ss) / 2
                dot_prop = (np.nansum(native_ss, axis=0) == 0).sum() / len(native_ss)
                if bps < 4 or dot_prop > 0.7:
                    # if self.skip_loops or np.random.rand() < dot_prop ** 2:
                    if self.skip_loops:
                        if self.warning:
                            warn = f'{pid} has too few base pairs! Skip this sample!'
                            logging.warning(warn)
                        return {}
            else:
                native_ss = None
        else:
            native_ss = None
        if self.return_coord:
            native_coord = {}
            for atom in coord_npz.keys():
                native_coord[atom] = torch.from_numpy(coord_npz[atom]).to(self.device)
            if torch.isnan(native_coord["C4'"]).all():
                if self.warning:
                    warn = f'{file} only have P! Skip this sample!'
                    logging.warning(warn)
                return {}
            if "N1/9" not in native_coord:
                native_coord["N1/9"] = np.where(((seq == 0) | (seq == 3))[0, :, None], native_coord["N9"],
                                                native_coord["N1"])

        if len(msa.shape) == 1:
            if self.warning:
                warn = f'{file} has an invalid msa shape {msa.shape}! Skip this sample!'
                logging.warning(warn)
            return {}
        elif len(msa.shape) == 3:
            msa = msa[0]

        if self.subsample_msa:
            msa = subsample(msa, limit=self.rowmax)
        elif msa.shape[0] > 20000:
            msa = msa[:self.rowmax]

        msa = self.to_torch(msa)
        sample = {
            'inter_labels': defaultdict(dict)
        }

        l = msa.shape[-1]
        try:
            inter_labels = {
                'distance': {atom: np.squeeze(npz[atom]) for atom in
                             ['P', "C3'", "C1'", "C4", "C4'", 'N1', 'CiNj', 'PiNj'] if atom in npz.files},
                'contact': {'all atom': np.squeeze(npz['contact'])}
            }
        except zlib.error:
            if self.warning:
                warn = f'{file} zlib err! Skip this sample!'
                logging.warning(warn)
            return {}
        except KeyError as e:
            if self.warning:
                warn = f'{file} key err: {str(e)}! Skip this sample!'
                logging.warning(warn)
            return {}

        if "C4'" not in npz.files:
            inter_labels['distance']["C4'"] = cdist(coord_npz["C4'"], coord_npz["C4'"])
        if pid.startswith('R11'):
            if len(inter_labels['contact']['all atom']) < l:
                for _ in range(l - len(inter_labels['contact']['all atom'])):
                    inter_labels['contact']['all atom'] = np.insert(inter_labels['contact']['all atom'], -1, np.nan,
                                                                    axis=0)
                    inter_labels['contact']['all atom'] = np.insert(inter_labels['contact']['all atom'], -1, np.nan,
                                                                    axis=1)
                    for atom in ['P', "C3'", "C1'", "C4", 'N1', 'CiNj', 'PiNj']:
                        inter_labels['distance'][atom] = np.insert(inter_labels['distance'][atom], -1, np.nan, axis=0)
                        inter_labels['distance'][atom] = np.insert(inter_labels['distance'][atom], -1, np.nan, axis=1)
        distances = copy.deepcopy(inter_labels['distance'])
        contact = np.squeeze(inter_labels['contact']['all atom'])

        if self.cont_for_missed_SS and self.return_nativess and native_ss is None:
            native_ss = copy.deepcopy(contact)
        if self.return_ss:
            ss = None
            if self.ss_dir is not None:
                ss_file = os.path.join(self.ss_dir, pid, 'bagged.npy')
                if os.path.isfile(ss_file):
                    ss = np.load(ss_file)
            else:
                ss = copy.deepcopy(native_ss)
                ss_noise = np.random.uniform(0, 1, size=ss.shape)
                ss = (ss + ss_noise) / 2
                for ss_name in self.ss_name.split(','):
                    if ss_name in npz.files:
                        ss = npz[ss_name]
                        break
            if ss is not None and len(ss.shape) == 3:
                ss = ss[0]
        return_ss = self.return_ss and ss is not None

        if return_ss:
            if (np.nanmax(np.triu(ss)) == 0 or np.nanmax(np.tril(ss).max()) == 0):
                ss += ss.T

        shape_er = False
        try:
            crop = self.get_crop(l, pid, npz, native_ss)
        except:
            if self.warning:
                warn = f'{pid} crop index error! Skip this sample!'
                logging.warning(warn)
            return {}
        if crop is None:
            if self.warning:
                warn = f'{pid} no contact! Skip this sample!'
                logging.warning(warn)
            return {}

        try:
            sample['inter_labels'] = self.parse_inter_labels(inter_labels, l, crop, binning=not pid.startswith('RF'),
                                                             obj_dict=obj)
        except Exception as e:
            if 'shape' in str(e):
                if self.warning:
                    warn = f'{pid} shape error! Skip this sample!'
                    logging.warning(warn)
            return {}
        msa = msa[:, crop]
        if return_ss:
            ss = ss[crop][:, crop]
        if native_ss is not None:
            native_ss = native_ss[crop][:, crop]

        if coord_npz is not None:
            for atom in native_coord:
                if isinstance(native_coord[atom], List):
                    try:
                        native_coord[atom] = torch.cat(native_coord[atom], dim=0)
                    except:
                        print(pid)
                native_coord[atom] = native_coord[atom][crop]
        if return_ss:
            if not msa.shape[1] == ss.shape[0]:
                if self.warning:
                    warn = f'{file} shape error: msa {msa.shape[1]} ss {ss.shape[0]}! Skip this sample!'
                    logging.warning(warn)
                shape_er = True
        if native_ss is not None:
            if not msa.shape[1] == native_ss.shape[0]:
                if self.warning:
                    warn = f'{file} shape error: msa {msa.shape[1]} native_ss {native_ss.shape[0]}! Skip this sample!'
                    logging.warning(warn)
                shape_er = True

        if shape_er:
            if self.warning:
                warn = f'{file} shape error! Skip this sample!'
                logging.warning(warn)
            return {}
        if torch.isnan(sample['inter_labels']['distance']["C3'"]).all():
            if self.warning:
                warn = f'{file} only containing P atoms! Skip this sample!'
                logging.warning(warn)
            return {}

        if 'res_id' in npz.files:
            sample['idx'] = npz['res_id'][crop].astype(np.int32)
        else:
            sample['idx'] = crop.astype(np.int32)
        sample['distance'] = {k: distances[k][crop][:, crop] for k in distances}
        sample['contact'] = sample['inter_labels']['contact']
        sample['msa'] = msa
        if self.return_coord:
            sample['coords'] = native_coord
        if return_ss:
            ss[np.isnan(ss)] = 0
            sample['ss'] = ss
        if native_ss is not None:
            sample['native_ss'] = native_ss
        sample['pid'] = pid
        if msa.shape[-1] <= 1:
            return {}
        if return_npz:
            return sample, npz
        return sample

    def get_crop(self, l, pid, npz, native_ss):
        idx = np.arange(l)

        if l > self.lengthmax:
            try:
                if pid.startswith('RF'):
                    pair_mask = (np.min([npz[atm][..., 1:10].sum(axis=-1) for atm in ["P", "C3'", "N1"]],
                                        axis=0) > 0.5)
                else:
                    pair_mask = (np.min([npz['P'], npz['N1'], npz['PiNj'], npz['PiNj'].T], axis=0) < 12)
                pair_mask = np.squeeze(pair_mask)
                if native_ss is not None:
                    pair_mask |= (native_ss > 0)
                if pair_mask.sum() == 0:
                    return
                valid_idx = idx[pair_mask.sum(axis=1) > 0]
                if self.random_:
                    point = random.choice(valid_idx)
                else:
                    point = valid_idx[0]
                crop = idx[pair_mask[point]]
                while len(crop) < self.lengthmax:
                    new_res = idx[(pair_mask[crop]).max(0)]
                    crop_new = np.sort(np.unique(np.concatenate([crop, new_res])))
                    if len(crop_new) == len(crop):
                        break
                    if len(crop_new) <= self.lengthmax:
                        crop = crop_new
                    else:
                        new_res = np.random.choice(new_res, self.lengthmax - len(crop), replace=False)
                        crop = np.sort(np.unique(np.concatenate([crop, new_res])))
                        break
            except ValueError:
                raise ValueError(f'{pid} value err when cropping!')
        else:
            crop = np.arange(l)

        return crop

    def parse_inter_labels(self, inter_labels, l, crop=None, obj_dict=obj, binning=True):
        if crop is None: crop = np.arange(l)
        labels = defaultdict(dict)
        for k in obj_dict['inter_labels']:
            for kk in obj_dict['inter_labels'][k]:
                label = inter_labels[k][kk]
                if not (l == label.shape[0]):
                    raise ValueError('shape err!')
                if k == 'contact' or not binning:
                    labels[k][kk] = self.to_torch(label[crop][:, crop])
                else:
                    binned = np.digitize(label[crop][:, crop], bins[k])
                    if k == 'distance':
                        binned[(binned >= n_bins['inter_labels']['distance']) & (~np.isnan(label[crop][:, crop]))] = 0
                        dist_binned = binned
                    else:
                        binned[dist_binned == 0] = 0
                    onehot = self.to_torch(np.arange(n_bins['inter_labels'][k]) == binned[..., None]).long()
                    labels[k][kk] = onehot
        return labels

    def to_torch(self, arr):
        return torch.from_numpy(arr).to(self.device)
