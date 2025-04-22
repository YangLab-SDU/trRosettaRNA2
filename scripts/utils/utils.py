import pandas as pd
import numpy as np
import pickle as pkl
import os
from pathlib import Path
#from scipy.io import savemat
#import scipy.stats

base_path = str(Path().resolve())

#print(str(base_path) + '/predictions/')

########################################################################################################
########################################################################################################
# ------------- one hot encoding of RNA sequences -----------------#
BASES = 'AUGC'

def one_hot(seq):
    RNN_seq = seq
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in RNN_seq])

    return feat

def z_mask(seq_len):
    mask = np.ones((seq_len, seq_len))
    return np.triu(mask, 1)


def l_mask(inp, seq_len, missing_nts):

	mask = np.ones((seq_len, seq_len))

	if len(missing_nts)>0:
		for i in missing_nts:
			mask[i,:] = 0
			mask[:,i] = 0

	return np.triu(mask, 1)

def get_data_final(args, seq, one_hot, profile, bp_prob, dca, missing_nts):

	seq_len = len(seq)
	zero_mask = z_mask(seq_len)[None, :, :, None]
	label_mask = l_mask(one_hot, seq_len, missing_nts)

	if args.single_seq == 1:

		temp = one_hot[None, :, :]
		temp = np.tile(temp, (temp.shape[1], 1, 1))
		feature = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2)

		feature = np.concatenate([feature, np.expand_dims(bp_prob, axis=2)], axis=2)
		assert feature.shape==(seq_len,seq_len, 9)
	else:
		profile_one_hot = np.concatenate([one_hot, profile], axis=1)

		temp = profile_one_hot[None, :, :]
		temp = np.tile(temp, (temp.shape[1], 1, 1))
		feature = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2)

		feature = np.concatenate([feature, np.expand_dims(bp_prob, axis=2), np.expand_dims(dca, axis=2)], axis=2)
		assert feature.shape==(seq_len,seq_len, 18)

	return seq_len, np.expand_dims(feature, axis=0), zero_mask, label_mask


def get_data(args, rna_id):

	with open(args.input_feats + '/' + rna_id) as f:
		temp_1 = pd.read_csv(f, comment='#', delim_whitespace=True, header=None, skiprows=[0]).values
	seq_ref = ''.join([j.upper() for j in temp_1[0, 0]])

	one_hot_feat = one_hot(seq_ref)

	if args.single_seq != 1:
		with open(args.input_feats + '/' + rna_id + '.pssm') as f:
			temp = pd.read_csv(f, comment='#', delim_whitespace=True, header=None).values

		profile = temp[:, 1:5].astype(float)
		off_set = np.zeros((len(seq_ref), profile.shape[1])) + 0.3

		for k, K in enumerate(seq_ref):
			try:
				off_set[k, BASES.index(K)] = 8.7
			except:
				pass

		profile += off_set
		profile /= np.sum(profile, axis=1, keepdims=True)
		profile = -np.log(profile)
	else:
		profile = []

#	profile_one_hot = np.concatenate([one_hot_feat, profile], axis=1)

############ read RNAfold base-pair probability output ##############################
#	with open(args.input_feats + '/' + rna_id + '.prob') as f:
#		temp = pd.read_csv(f, comment='#', header=None).values
	with open(args.input_feats + '/' + rna_id + '_dp.ps') as f:
		temp = pd.read_csv(f, comment='%', header=None)
	temp = [i for i in temp[0] if i.split(' ')[-1]=='ubox']
	bp_prob_rnafold = np.zeros((len(seq_ref), len(seq_ref)))
	for i in temp:
		a = i.split(' ')
		bp_prob_rnafold[int(a[0]) - 1, int(a[1]) - 1] = float(a[2])**2

	if args.single_seq != 1:
	################ read plmc output ##############################
		with open(args.input_feats + '/' + rna_id + '.dca') as f:
			temp4 = pd.read_csv(f, comment='#', delim_whitespace=True, header=None, usecols=[0,2,5]).values
		dca2 = np.zeros((len(seq_ref), len(seq_ref)))
		for k in temp4:
			if abs(int(k[0]) - int(k[1])) < 4:
				dca2[int(k[0]-1), int(k[1]-1)] = 1*k[2]
			else:
				dca2[int(k[0]-1), int(k[1]-1)] = k[2]
	else:
		dca2 = []

	seq_len, feature, zero_mask, label_mask = get_data_final(args, seq_ref, one_hot_feat, profile, bp_prob=bp_prob_rnafold, dca=dca2, missing_nts=[])

	return seq_ref, seq_len, feature, zero_mask, label_mask

def prob_to_secondary_structure(ensemble_outputs, label_mask, seq, name, args, base_path):

    test_output = ensemble_outputs

    inds = np.where(label_mask == 1)
    y_pred = np.zeros(label_mask.shape)
    for i in range(test_output.shape[0]):
        y_pred[inds[0][i], inds[1][i]] = test_output[i]

    if args.outputs=='outputs/':  output_path = os.path.join(base_path, args.outputs)
    else: output_path = args.outputs

    if args.single_seq == 1: np.savetxt(output_path + '/'+ name +'.prob_single', y_pred, delimiter='\t')
    else: np.savetxt(output_path + '/'+ name +'.prob_profile', y_pred, delimiter='\t')

    return
