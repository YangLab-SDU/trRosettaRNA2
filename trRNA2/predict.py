import sys, os
import tempfile

from argparse import ArgumentParser

from .utils import *
from .model_ss import SSpredictor
from .model_3d import Folding

parser = ArgumentParser()
parser.add_argument('-i',
                    '--msa',
                    help='(required) input MSA file')
parser.add_argument('-o',
                    '--out_dir',
                    help='(required) output directory')
parser.add_argument('-mdir', '--model_pth',
                    default=f'params/bench2022',
                    help='pretrained params directory')
parser.add_argument('-mname',
                    '--model_name',
                    default='model_1',
                    help='model name')
parser.add_argument('-ss',
                    '--ss_file',
                    default=None,
                    help='the custom secondary structure (SS) file; using trRNA2-SS if not provided (dafault).')
parser.add_argument('-ss_fmt',
                    '--ss_fmt',
                    default='dot_bracket',
                    choices=['dot_bracket', 'ct', 'bpseq', 'prob', ],
                    help='the format of custom SS file; dot_bracket(default)/ct/bpseq/prob')
parser.add_argument('-nrows',
                    '--nrows',
                    default=500, type=int,
                    help='maximum number of rows in the MSA repr (default: 500).')
parser.add_argument('-refine_steps',
                    '--refine_steps',
                    default=200, type=int,
                    help='maximum steps of refinement (default: 200).')
parser.add_argument('-mid',
                    '--return_mid',
                    action='store_true',
                    help='whether return mid-cycle predictions (default: False).')
parser.add_argument('-gpu',
                    '--gpu',
                    default=0,
                    type=int,
                    help='use which gpu')
parser.add_argument('-cpu',
                    '--cpu',
                    default=5, type=int,
                    help='number of CPUs to use')

group = parser.add_argument_group('(Optional) Arguments for PyRosetta version')
group.add_argument('-pyrosetta',
                   '--pyrosetta',
                   action='store_true',
                   help='whether run energy minimization (i.e., PyRosetta version; default: False).')
group.add_argument('-fas',
                   '--fas',
                   default=None, type=str,
                   help='input FASTA file')
group.add_argument('-nm',
                   '--nmodels',
                   default=5, type=int,
                   help='number of decoys to generate')
group.add_argument('-dcut',
                   '--dcut',
                   default=0.45, type=float,
                   help='cutoff of distance restraints')
group.add_argument('-tmp',
                   '--tmpdir',
                   default='/dev/shm/',
                   help='temp folder to store all the restraints')
args = parser.parse_args()


def predict(model, seq, msa, ss):
    with torch.no_grad():
        L = len(seq)
        res_id = torch.arange(L, device=device).view(1, L)

        if ss is not None:
            ss = ss.squeeze().to(device)
            if not (ss.shape[-1] == msa.shape[-1] == L):
                raise ValueError(
                    f'Length mismatch: seq length {L}, ss length {ss.shape[-1]}, msa length {msa.shape[-1]}!')
            ss = ss.view(1, L, L)
        msa = msa.view(1, -1, L)
        outputs_all, outputs = model(seq, msa, ss, res_id=res_id.to(device),
                                     msa_cutoff=args.nrows, config=config)

    outputs_tosave_all = {}
    for c in outputs_all:
        outputs = outputs_all[c]
        outputs_tosave = {}
        for k in outputs:
            if isinstance(outputs[k], torch.Tensor):
                outputs_tosave[k] = outputs[k].cpu().detach().numpy()
            elif k == 'frames':
                outputs_tosave[k] = {
                    'R': outputs[k][0].cpu().detach().numpy(), 't': outputs[k][1].cpu().detach().numpy()
                }
            elif k == 'frames_allatm':
                outputs_tosave[k] = {
                    fname: {'R': tup[0].cpu().detach().numpy(), 't': tup[1].cpu().detach().numpy()}
                    for fname, tup in outputs[k].items()
                }
            elif k == 'geoms':
                pred_dict = outputs['geoms']['inter_labels']
                for kk in pred_dict:
                    for kkk in pred_dict[kk]:
                        pred_dict[kk][kkk] = pred_dict[kk][kkk].cpu().detach().numpy()
                outputs_tosave['inter_labels'] = pred_dict
        outputs_tosave_all[c] = outputs_tosave

    return outputs_tosave_all, outputs_all


if __name__ == '__main__':

    if args.pyrosetta:
        assert args.fas is not None, 'please specify --fas if PyRosetta version is needed!'
        args.refine_steps = 0

    torch.set_num_threads(args.cpu)
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')
    py = sys.executable

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print('---------Load model and config---------')
    config = read_json(f'{args.model_pth}/config/{args.model_name}.json')
    structure_module_config = config['structure_module']
    model_ckpt = torch.load(f'{args.model_pth}/models/{args.model_name}.pth.tar', map_location=device,
                            weights_only=True)
    model_ckpt = {k: model_ckpt[k] for k in model_ckpt if 'refinenet' not in k}
    if "to_dist.fc_2d.distance.C4'.1.weight" in model_ckpt:
        obj['inter_labels']['distance'] = ["C3'", "P", "N1", "C4", "C1'", "CiNj", "PiNj", "C4'"]
    model = Folding(dim_2d=config['dim_pair'], layers_2d=config['RNAformer']['n_block'], config=config).to(device)
    model.load_state_dict(model_ckpt)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print('------------Read input file------------')
    cwd = os.getcwd()
    msa = parse_a3m(args.msa, limit=20000)
    msa = torch.from_numpy(msa).to(device)[None]

    raw_seq = open(args.msa).readlines()[1].strip().replace('T', 'U').replace('-', '')

    if args.ss_file is None:
        print('----------------Predict----------------')
        ss_models = []
        for nm in range(1, 4):
            ss_mname = f'model_{nm}_finetune'
            config_ss = read_json(f'{args.model_pth}/config_ss/{ss_mname}.json')

            ss_model = SSpredictor(dim_2d=config_ss['dim_pair'],
                                   layers_2d=config_ss['RNAformer']['n_block'],
                                   config=config_ss, device=device,
                                   ).to(device)
            model_ckpt = torch.load(f'{args.model_pth}/models_ss/{ss_mname}.pth.tar', map_location=device,
                                    weights_only=True)
            ss_model.load_state_dict(model_ckpt)
            ss_models.append(ss_model)

        with torch.no_grad():
            ss_lst = []
            for ss_model in ss_models:
                ss = ss_model(msa).float()
                ss_lst.append(ss)
            ss = torch.mean(torch.stack(ss_lst, dim=0), dim=0)
    else:
        if config['use_ss']:
            if args.ss_fmt == 'dot_bracket':
                ss = ss2mat(open(args.ss_file).read().rstrip().splitlines()[-1].strip())
            elif args.ss_fmt == 'ct':
                ss = parse_ct(args.ss_file, length=len(msa[0]))
            elif args.ss_fmt == 'bpseq':
                ss = parse_bpseq(args.ss_file)
            elif args.ss_fmt == 'prob':
                ss = np.loadtxt(args.ss_file)
                ss += ss.T
            ss = torch.from_numpy(ss).float()
        else:
            ss = None
        print('----------------Predict----------------')

    outputs_tosave_all, outputs_all = predict(model, raw_seq, msa, ss)

    unrelaxed_model = os.path.abspath(f'{out_dir}/model_1.pdb')
    refined_model = os.path.abspath(f'{out_dir}/model_1_refined{args.refine_steps}.pdb')
    npz = os.path.abspath(f'{out_dir}/model_1_2D.npz')

    for c in outputs_tosave_all:
        outputs = outputs_all[c]
        outputs_tosave = outputs_tosave_all[c]

        node_cords_pred = outputs['cord_tns_pred'][-1].squeeze(0).permute(1, 0, 2)
        chain_id = 'A'
        save_pdb = unrelaxed_model.replace('.pdb', f'_c{c}.pdb') if c < max(outputs_tosave_all) else unrelaxed_model

        model.structure_module.converter.export_pdb_file(raw_seq,
                                                         node_cords_pred.data.cpu().numpy(),
                                                         path=save_pdb, chain_id=chain_id,
                                                         confidence=outputs['plddt'][0].data.cpu().numpy(),
                                                         )
        npz_dict = outputs_tosave['inter_labels']
        npz_dict['plddt'] = outputs['plddt'][-1].data.cpu().numpy()
        np.savez_compressed(npz.replace('.npz', f'_c{c}.npz') if c < config['max_recycle'] else npz, **npz_dict, )

    if args.refine_steps is not None and args.refine_steps > 0:
        print('refining......')
        from .folding.refine import refine

        refine(unrelaxed_model, args.refine_steps)

    if args.pyrosetta:
        print('running pyrosetta...')
        from .folding.utils_cst import npz2cst
        from .folding.utils_ros import fold_all

        tmpdir = tempfile.TemporaryDirectory(prefix=args.tmpdir + '/')
        args.tmpdir = tmpdir.name
        print('temp folder:     ', args.tmpdir)
        # parse npz into rosetta-format restraint files
        npz2cst(args, geoms=outputs_tosave['inter_labels'])

        # perform energy minimization
        fold_all(args, out_pdb=f'{out_dir}/model_1_pyrosetta.pdb')

    print('done!')
    plddt_global = npz_dict['plddt'].mean()
    print(f'pLDDT: {plddt_global:.3f}')
