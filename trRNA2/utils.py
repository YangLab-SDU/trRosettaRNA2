import json
import string
import torch
import numpy as np

from einops.layers.torch import Rearrange
from torch import nn

obj = {
    'inter_labels': {
        'distance': ["C3'", "P", "N1", "C4", "C1'", "CiNj", "PiNj"],
        'contact': ['all atom']
    }
}

bins_size = 1
dist_max = 40
dist_min = 3

n_bins = {
    'intra_labels': {
        'bond_length': 1,
        'bond_angle': 13,
        'dihedral_angle': 25,
    },
    'inter_labels': {
        'distance': int((dist_max - dist_min) / bins_size + 1),
        'angle': 13,
        'dihedral_angle': 25,
        'contact': 1
    }
}
bins = {
    'distance': np.linspace(dist_min, dist_max, int((dist_max - dist_min) / bins_size + 1)),
    'angle': np.linspace(0.0, np.pi, 13),
    'dihedral_angle': np.linspace(-np.pi, np.pi, 25),
    'bond_angle': np.linspace(70, 160, int((160 - 70) / 7.5 + 1)) * np.pi / 180
}


def one_hot(x, bin_values=torch.arange(3, 40.5, 1)):
    """ 0~2,2~2.5,...,19.5~20"""
    bin_values = bin_values.to(x.device)
    n_bins = len(bin_values)
    bin_values = bin_values.view([1] * x.ndim + [-1])
    binned = (bin_values <= x[..., None]).sum(-1)
    binned = torch.where(binned > n_bins - 1, n_bins - 1, binned)
    onehot = (torch.arange(n_bins, device=x.device) == binned[..., None]).float()
    return onehot


def parse_a3m(filename, limit=20000, rm_query_gap=True):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    n = 0
    for line in open(filename, "r"):
        if line[0] != '>' and len(line.strip()) > 0:
            seqs.append(
                line.rstrip().replace('W', 'A').replace('R', 'A').replace('Y', 'C').replace('E', 'A').replace('I',
                                                                                                              'A').replace(
                    'P', 'G').replace('T', 'U').translate(table))
            n += 1
            if n == limit:
                break

    # convert letters into numbers
    alphabet = np.array(list("AUCGTN-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i if i != 4 else 1
    if rm_query_gap:
        msa = msa[:, msa[0] != 6]
    # treat all unknown characters as gaps
    msa[msa > 4] = 4

    return msa


def ss2mat(ss_seq):
    ss_mat = np.zeros((len(ss_seq), len(ss_seq)))
    stack = []
    stack1 = []
    stack2 = []
    stack3 = []
    stack_alpha = {alpha: [] for alpha in string.ascii_lowercase}
    for i, s in enumerate(ss_seq):
        if s == '(':
            stack.append(i)
        elif s == ')':
            ss_mat[i, stack.pop()] = 1
        elif s == '[':
            stack1.append(i)
        elif s == ']':
            ss_mat[i, stack1.pop()] = 1
        elif s == '{':
            stack2.append(i)
        elif s == '}':
            ss_mat[i, stack2.pop()] = 1
        elif s == '<':
            stack3.append(i)
        elif s == '>':
            ss_mat[i, stack3.pop()] = 1
        elif s.isalpha() and s.isupper():
            stack_alpha[s.lower()].append(i)
        elif s.isalpha() and s.islower():
            ss_mat[i, stack_alpha[s].pop()] = 1
        elif s in ['.', ',', '_', ':', '-']:
            continue
        else:
            raise ValueError(f'unk not: {s}!')
    allstacks = stack + stack1 + stack2 + stack3
    for _, stack in stack_alpha.items():
        allstacks += stack
    if len(allstacks) > 0:
        raise ValueError('Provided dot-bracket notation is not completely matched!')

    ss_mat += ss_mat.T
    return ss_mat


def parse_ct(ct_file, length=None):
    seq_ct = ''
    if length is None:
        length = int(open(ct_file).readlines()[0].split()[0])
    mat = np.zeros((length, length))
    for line in open(ct_file):
        items = line.split()
        if len(items) >= 6 and items[0].isnumeric() and items[2].isnumeric() and items[3].isnumeric() and items[
            4].isnumeric():
            seq_ct += items[1]
            if int(items[4]) > 0:
                mat[int(items[4]) - 1, int(items[5]) - 1] = 1
                mat[int(items[5]) - 1, int(items[4]) - 1] = 1
    return mat


def parse_bpseq(bpseq_file):
    lines = [line for line in open(bpseq_file).readlines() if not line.startswith('#')]
    len_ = len(lines)
    ss_mat = np.zeros((len_, len_))
    seq = ''
    for line in lines:
        aa = line.split()[1]
        seq += aa
        i = int(line.split()[0])
        j = int(line.split()[-1])
        if j > 0:
            ss_mat[i - 1, j - 1] = 1
    return ss_mat


def save_to_json(obj, file):
    with open(file, "w") as f:
        jso = json.dumps(obj, cls=NpEncoder)
        f.write(jso)


def read_json(file):
    with open(file, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class Symm(nn.Module):
    def __init__(self, pattern):
        super(Symm, self).__init__()
        self.pattern = pattern

    def forward(self, x):
        return (x + Rearrange(self.pattern)(x)) / 2
