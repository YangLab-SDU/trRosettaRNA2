#!/usr/bin/env /usr/bin/python
import argparse
import numpy as np

from pyrosetta import *
import concurrent.futures
from pyrosetta.rosetta import *
from pyrosetta.rosetta.core.pose.rna import *
from pyrosetta.rosetta.core.scoring import ScoreFunction, ScoreType
from pyrosetta.rosetta.protocols.constraint_movers import ConstraintSetMover
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.pose import *
import re, sys
import os, shutil
import math
import random


def randTrial(your_pose):
    randNum = random.randint(2, your_pose.total_residue())

    curralpha = your_pose.alpha(randNum)
    currbeta = your_pose.beta(randNum)
    currgamma = your_pose.gamma(randNum)
    currdelta = your_pose.delta(randNum)
    currepsilon = your_pose.epsilon(randNum)
    currzeta = your_pose.zeta(randNum)
    currchi = your_pose.chi(randNum)

    newalpha = random.gauss(curralpha, 25)
    newbeta = random.gauss(currbeta, 25)
    newgamma = random.gauss(currgamma, 25)
    newdelta = random.gauss(currdelta, 25)
    newepsilon = random.gauss(currepsilon, 25)
    newzeta = random.gauss(currzeta, 25)
    newchi = random.gauss(currchi, 25)

    your_pose.set_alpha(randNum, newalpha)
    your_pose.set_beta(randNum, newbeta)
    your_pose.set_gamma(randNum, newgamma)
    your_pose.set_delta(randNum, newdelta)
    your_pose.set_epsilon(randNum, newepsilon)
    your_pose.set_zeta(randNum, newzeta)
    your_pose.set_chi(randNum, newchi)

    return your_pose


def score(your_pose):
    sf = rna_lowres_sf(your_pose)
    return sf


def decision(before_pose, after_pose):
    E = score(after_pose) - score(before_pose)
    if E < 0:
        return after_pose
    elif random.uniform(0, 1) >= math.exp(-E / 1):
        return before_pose
    else:
        return after_pose


def basic_folding(your_pose):
    lowest_pose = Pose()  # Create an empty pose for tracking the lowest energy pose.
    for i in range(120):
        if i == 0:
            lowest_pose.assign(your_pose)

        before_pose = Pose()
        before_pose.assign(your_pose)  # keep track of pose before random move

        after_pose = Pose()
        after_pose.assign(randTrial(your_pose))  # do rand move and store the pose

        your_pose.assign(decision(before_pose, after_pose))  # keep the new pose or old pose

        if score(your_pose) < score(lowest_pose):  # updating lowest pose
            lowest_pose.assign(your_pose)

    return lowest_pose


def generate_start_model(seq):
    assembler = core.import_pose.RNA_HelixAssembler()
    # print(seq)
    initpose = assembler.build_init_pose(seq, '')  # helix pose
    initpose.dump_pdb("init.pdb")
    pose = basic_folding(initpose)
    pose.remove_constraints()
    pose.dump_pdb("init_basic.pdb")
    return pose


def fold_with_cst(cst_file, seq, pose, max_iter, repeat_times):
    mmap = MoveMap()
    mmap.set_bb(True)  ##Whether the frame dihedral angle changes
    mmap.set_chi(True)  ##Whether the dihedral angle of the side chain changes
    mmap.set_jump(True)  ##Relative movement between polypeptide chains
    min_mover = rosetta.protocols.minimization_packing.MinMover(mmap, op_score, 'lbfgs_armijo_nonmonotone', 0.0001,
                                                                True)
    min_mover.max_iter(max_iter)
    repeat_mover = RepeatMover(min_mover, repeat_times)

    add_cst(pose, cst_file)
    # repeat_mover.apply(pose)

    mover = min_mover
    lowest_pose = pose
    score = float(op_score(pose))
    print("energy of initial model:", score)
    for i in range(repeat_times):
        mover.apply(pose)
        score1 = float(op_score(pose))
        print("energy of model", i, score1)
        if (score1 < score):
            lowest_pose.assign(pose)
            score = score1

    pose = lowest_pose
    return pose, score


def read_fasta(file):
    fasta = "";
    with open(file, "r") as f:
        for line in f:
            if (line[0] == ">"):
                continue
            else:
                line = line.rstrip()
                fasta = fasta + line;
    return fasta


def read_cst(file):
    array = []
    with open(file, "r") as f:
        for line in f:
            line = line.rstrip()
            array.append(line)
    return array


def add_cst(pose, cstfile):
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(cstfile)
    constraints.add_constraints(True)
    constraints.apply(pose)


def remove_clash(scorefxn, mover, pose):
    clash_score = float(scorefxn(pose))
    if (clash_score > 10):
        for nm in range(0, 10):
            mover.apply(pose)
            clash_score = float(scorefxn(pose))
            if (clash_score < 10): break


def fetch_cst(cst, sep1, sep2, cut, std_cut=None):
    array = []
    for line in cst:
        # print(line)
        line = line.rstrip()
        b = line.split()
        cst_name = b[0]
        pcut = cut[cst_name]

        if std_cut is not None:
            if not line.endswith('#cont'):
                p_std = float(line.split('#')[1].split()[0])
            else:
                p_std = 10000
        if line.endswith('#cont'):
            prob = 1
        else:
            prob = float(b[-1])

        i = int(b[2])
        j = int(b[4])

        sep = abs(j - i)
        if (sep < sep1 or sep >= sep2): continue
        if std_cut is not None and p_std < std_cut: continue
        if (prob >= pcut):
            array.append(line)
    return array


# def run_min(array, pose, mover, tmpname):
# add_cst(pose, array, tmpname)
# mover.apply(pose)

def run_min(cstfile, pose, mover1, mover2=None):
    add_cst(pose, cstfile)
    mover1.apply(pose)
    if mover2 is not None:
        mover2.apply(pose)


def run_refine(pose):
    idealize = rosetta.protocols.idealize.IdealizeMover()
    poslist = rosetta.utility.vector1_unsigned_long()

    scorefxn = create_score_function('empty')
    scorefxn.set_weight(rosetta.core.scoring.cart_bonded, 1.0)
    scorefxn.score(pose)

    emap = pose.energies()
    # print("idealize...")
    for res in range(1, len(pose.residues) + 1):
        cart = emap.residue_total_energy(res)
        if cart > 50:
            poslist.append(res)
            # print("idealize %d %8.3f" % (res, cart))

    if len(poslist) > 0:
        idealize.set_pos_list(poslist)
    try:
        idealize.apply(pose)

        # cart-minimize
        scorefxn_min = create_score_function('ref2015_cart')

        mmap = MoveMap()
        mmap.set_bb(True)
        mmap.set_chi(True)
        mmap.set_jump(True)
        mmap.set_chi(False)

        min_mover = rosetta.protocols.minimization_packing.MinMover(mmap, scorefxn_min, 'lbfgs_armijo_nonmonotone',
                                                                    0.00001, True)
        # min_mover = rosetta.protocols.minimization_packing.MinMover(mmap, op_score, 'lbfgs_armijo_nonmonotone', 0.00001, True)
        min_mover.max_iter(200)
        min_mover.cartesian(True)
        # print("minimize...")
        min_mover.apply(pose)

    except:
        print('!!! idealization failed !!!')


def fold_single(args):
    mmap = MoveMap()
    mmap.set_bb(True)  ##Whether the frame dihedral angle changes
    mmap.set_chi(True)  ##Whether the dihedral angle of the side chain changes
    mmap.set_jump(True)  ##Relative movement between polypeptide chains
    clash_mover = rosetta.protocols.minimization_packing.MinMover(mmap, scorefxn_clash, 'lbfgs_armijo_nonmonotone',
                                                                  0.0001, True)
    clash_mover.max_iter(200)

    pose = generate_start_model(seq.lower())

    # step 1. fold with high-confidence restraints
    print("fold with high-confidence restraints")
    cstfile = f'{args.tmpdir}/cstfile_high.txt'
    pose, E = fold_with_cst(cstfile, seq.lower(), pose, max_iter, 3)
    it = 0
    pose1 = pose
    while (E > 500):
        it = it + 1
        pose.remove_constraints()
        # pose=generate_start_model(seq.lower())
        pose, E1 = fold_with_cst(cstfile, seq.lower(), pose, max_iter, 2)
        op_score.show(pose)
        if (E1 < E):
            pose1 = pose
            E = E1
        if (it > 5): break
    pose = pose1
    op_score.show(pose)
    # name = "model_high_" + str(args.i) + ".pdb"
    # pose.dump_pdb(f'{args.tmp}/{name}')

    if (nres < 500):
        # step 2. with all cst
        pose.remove_constraints()
        cstfile = f'{args.tmpdir}/cstfile_all.txt'
        print("fold with all cst ")
        fold_with_cst(cstfile, seq.lower(), pose, max_iter, 3)
        op_score.show(pose)

        # name = "model_all_cst_" + str(args.i) + ".pdb"
        # pose.dump_pdb(f'{args.tmp}/{name}')

    # step 3. refine
    print("refine pose")
    run_refine(pose)

    # name = "ref_model_" + str(args.dcut) + "_" + str(args.i) + ".pdb"
    # pose.dump_pdb(f'{args.tmp}/{name}')
    # inf = name + "\t" + "opscore:" + str(op_score(pose))
    # SS = open(args.scoreout, "a")
    # SS.write(inf)
    # SS.write("\n")
    # SS.close()

    return pose


def fold_all(args, out_pdb):
    global rna_lowres_sf
    init(
        '-mute all -hb_cen_soft  -relax:dualspace true -relax:default_repeats 3 -default_max_cycles 200 -detect_disulf -detect_disulf_tolerance 3.0')

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    rna_lowres_sf = core.scoring.ScoreFunctionFactory.create_score_function("rna/denovo/rna_lores_with_rnp_aug.wts")

    global op_score, scorefxn_clash, seq, max_iter, nres
    seq = read_fasta(args.fas).replace('T', 'U').replace('N', 'A').lower()
    nres = len(seq)

    scorefxn_clash = create_score_function('ref2015')
    scorefxn_clash.set_weight(rosetta.core.scoring.fa_rep, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.atom_pair_constraint, 0.5)
    scorefxn_clash.set_weight(rosetta.core.scoring.fa_intra_rep, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.rna_base_pair, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.rna_base_stack, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.rna_torsion, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.suiteness_bonus, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.rna_sugar_close, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.fa_stack, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.rna_repulsive, 1.0)

    op_score = create_score_function('ref2015')
    op_score.set_weight(rosetta.core.scoring.atom_pair_constraint, 10.0)
    op_score.set_weight(rosetta.core.scoring.fa_rep, 10.0)
    op_score.set_weight(rosetta.core.scoring.fa_intra_rep, 9.0)
    op_score.set_weight(rosetta.core.scoring.rna_base_pair, 9.0)
    op_score.set_weight(rosetta.core.scoring.rna_base_stack, 9.0)
    op_score.set_weight(rosetta.core.scoring.rna_torsion, 1.0)
    op_score.set_weight(rosetta.core.scoring.suiteness_bonus, 1.0)
    op_score.set_weight(rosetta.core.scoring.rna_sugar_close, 1.0)
    op_score.set_weight(rosetta.core.scoring.fa_stack, 1.0)
    op_score.set_weight(rosetta.core.scoring.rna_repulsive, 10.0)

    scorefxn_clash = create_score_function('ref2015')
    scorefxn_clash.set_weight(rosetta.core.scoring.fa_rep, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.atom_pair_constraint, 0.5)
    scorefxn_clash.set_weight(rosetta.core.scoring.fa_intra_rep, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.rna_base_pair, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.rna_base_stack, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.rna_torsion, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.suiteness_bonus, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.rna_sugar_close, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.fa_stack, 1.0)
    scorefxn_clash.set_weight(rosetta.core.scoring.rna_repulsive, 1.0)

    cstpath = args.tmpdir + f'/cstfile_dist.txt'
    cstpath_ss = f'{args.tmpdir}/cstfile_ss.txt'

    global allcst_ss
    allcst_dist = read_cst(cstpath)
    # allcst_cont = read_cst(cstpath_cont)
    if (os.path.exists(cstpath_ss)):
        allcst_ss = read_cst(cstpath_ss)
        cst_op = allcst_dist + allcst_ss
    else:
        cst_op = allcst_dist

    mmap = MoveMap()
    mmap.set_bb(True)  ##Whether the frame dihedral angle changes
    mmap.set_chi(True)  ##Whether the dihedral angle of the side chain changes
    mmap.set_jump(True)  ##Relative movement between polypeptide chains
    clash_mover = rosetta.protocols.minimization_packing.MinMover(mmap, scorefxn_clash, 'lbfgs_armijo_nonmonotone',
                                                                  0.0001, True)
    clash_mover.max_iter(200)

    max_iter = nres * 30
    if (max_iter < 10000): max_iter = 10000

    # using all cst
    sep1 = 1
    sep2 = 10000

    minstd = 0.01
    std_cut = minstd  # +0.01*pcut['AtomPair']

    # step 1. generate model with ss potential
    pcut = {
        'AtomPair': 0.9,
    }

    cst_all = fetch_cst(cst_op, sep1, sep2, pcut, std_cut)
    n_rst = 0
    cstfile = f'{args.tmpdir}/cstfile_high.txt'
    F = open(cstfile, "w")
    for a in cst_all:
        n_rst = n_rst + 1
        F.write(a)
        F.write("\n")
    F.close()

    cst_all = cst_op
    cstfile = f'{args.tmpdir}/cstfile_all.txt'
    F = open(cstfile, "w")
    for a in cst_all:
        F.write(a)
        F.write("\n")
    F.close()

    executor = concurrent.futures.ProcessPoolExecutor(args.cpu)
    futures = [executor.submit(fold_single, args) for _ in range(args.nmodels)]
    results = concurrent.futures.wait(futures)
    poses = list(results[0])

    min_energy = np.inf
    for pose in poses:
        pose = pose.result()
        energy = op_score(pose)
        if energy < min_energy:
            best_pose = pose
            min_energy = energy

    best_pose.dump_pdb(out_pdb)
    print('\ndone')
