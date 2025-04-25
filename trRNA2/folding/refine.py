import os
import math
import random
from pyrosetta import *

init(' -relax:default_repeats 5 -mute all')

from pyrosetta.rosetta import *
from pyrosetta.rosetta.core.pose import *

os.environ["OPENBLAS_NUM_THREADS"] = "1"

rna_lowres_sf = core.scoring.ScoreFunctionFactory.create_score_function("rna/denovo/rna_lores_with_rnp_aug.wts")
rna_hires_sf = core.scoring.ScoreFunctionFactory.create_score_function("stepwise/rna/rna_res_level_energy4.wts")
fullatom_sf = create_score_function("ref2015")


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


def score1(your_pose):
    sf = rna_hires_sf(your_pose)
    return sf


def score2(your_pose):
    sf = create_score_function("ref2015")
    return sf(your_pose)


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


def fetch_cst_atomset(cst, atoms):
    array = []
    for line in cst:
        line = line.rstrip()
        toks = line.split()
        cst_atom = toks[1]
        if cst_atom not in atoms:
            # print(f'{cst_atom} ?????????')
            continue
        else:
            array.append(line)
    return array


def fetch_cst_atom(cst, atom):
    array = []
    for line in cst:
        line = line.rstrip()
        toks = line.split()
        cst_type = toks[0]
        cst_atom = toks[1]
        if cst_type == 'AtomPair' and (cst_atom == atom or (atom[0] == 'N' and cst_atom[0] == 'N')):
            array.append(line)
    return array


def fetch_cst_dihedral(cst):
    array = []
    for line in cst:
        cst_type = line.split()[0]
        if cst_type == 'Dihedral':
            array.append(line)
    return array


def add_cst(pose, array, tmpname):
    F = open(tmpname, "w")
    for a in array:
        F.write(a)
        # F.write(a.replace('fengcj','wangwk'))
        F.write("\n")
    F.close()
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(tmpname)
    constraints.add_constraints(True)
    constraints.apply(pose)
    os.remove(tmpname)


def remove_clash(scorefxn, mover, pose):
    clash_score = float(scorefxn(pose))
    if (clash_score > 10):
        for nm in range(0, 10):
            mover.apply(pose)
            clash_score = float(scorefxn(pose))
            if (clash_score < 10): break


def fetch_cst(cst, nres, sep1, sep2, cut, std_cut=None):
    array = []
    for line in cst:
        # print(line)
        line = line.rstrip()
        b = line.split()
        if std_cut is not None:
            if not line.endswith('#cont'):
                p_std = float(line.split('#')[1].split()[0])
            else:
                p_std = 10000
        if line.endswith('#cont'):
            prob = 1
        else:
            prob = float(b[-1])
        cst_name = b[0]
        pcut = cut[cst_name]

        if cst_name == 'AtomPair':
            i = int(b[2])
            j = int(b[4])
        elif cst_name == 'Angle':
            if b[1] == "C4'":  # CiNiCj
                j = int(b[6])
            elif b[1] in ['N1', "N9"]:  # NiCjPj+1
                j = int(b[4])
        elif cst_name == 'Dihedral':
            if b[1] == "P":  # Pj+1CjNiCi
                i = int(b[6])
                j = int(b[4])
            elif b[1] in ['N1', "N9"]:  # CjNiCiPi+1
                i = int(b[2])
                j = int(b[4])
            elif b[1] == "C4'":  # NiCjPj+1Cj+1
                i = int(b[4])
                j = int(b[2])
        if not line.endswith('#1d'):
            sep = abs(j - i)
            if (sep < sep1 or sep >= sep2): continue
        if std_cut is not None and p_std < std_cut: continue
        if (prob >= pcut):
            array.append(line)
    return array


# def run_min(array, pose, mover, tmpname):
# add_cst(pose, array, tmpname)
# mover.apply(pose)

def run_min(cst_all, n_sets, pose, mover1, mover2=None, tmpname=None):
    if (len(cst_all) == 0):

        print("warning: empty constraint set")
        mover1.apply(pose)
        if mover2 is not None:
            mover2.apply(pose)
        return

    random.shuffle(cst_all)
    b_size = int(len(cst_all) / n_sets)
    for i in range(0, len(cst_all), b_size):
        batch = cst_all[i:i + b_size]
        add_cst(pose, batch, tmpname)
        mover1.apply(pose)
        if mover2 is not None:
            mover2.apply(pose)


def refine(init_pdb, max_iter=200):
    op_score = create_score_function('ref2015')
    op_score.set_weight(rosetta.core.scoring.atom_pair_constraint, 9)
    op_score.set_weight(rosetta.core.scoring.dihedral_constraint, 4.0)
    op_score.set_weight(rosetta.core.scoring.angle_constraint, 4.0)
    op_score.set_weight(rosetta.core.scoring.fa_rep, 2)
    op_score.set_weight(rosetta.core.scoring.fa_stack, 1)
    op_score.set_weight(rosetta.core.scoring.stack_elec, 1)
    op_score.set_weight(rosetta.core.scoring.fa_intra_rep, 5)

    pose = pose_from_file(init_pdb)
    try:
        op_score.show(pose)
    except RuntimeError:
        new_lines = []
        for line in open(init_pdb):
            if (line.startswith('ATOM')
                    and line[19] == 'N'  # unknown AA; 'N' for RNA
            ):
                continue
            new_lines.append(line)
        with open(init_pdb.replace('.pdb', '_torefine.pdb'), 'w') as f:
            f.write(''.join(new_lines))
        pose = pose_from_file(init_pdb.replace('.pdb', '_torefine.pdb'))
        op_score.show(pose)

    idealize = rosetta.protocols.idealize.IdealizeMover()
    poslist = rosetta.utility.vector1_unsigned_long()

    scorefxn = create_score_function('empty')
    scorefxn.set_weight(rosetta.core.scoring.cart_bonded, 1.0)
    scorefxn.score(pose)

    # idealize the bond with high energy
    emap = pose.energies()
    print("idealize...")
    for res in range(1, len(pose.residues) + 1):
        cart = emap.residue_total_energy(res)
        if cart > 50:
            poslist.append(res)
            print("idealize %d %8.3f" % (res, cart))

    if len(poslist) > 0:
        idealize.set_pos_list(poslist)
    try:
        idealize.apply(pose)

        # cart-minimize to reduce clash
        scorefxn_min = create_score_function('ref2015_cart')

        mmap = MoveMap()
        mmap.set_bb(True)
        mmap.set_chi(True)
        mmap.set_jump(True)
        mmap.set_chi(False)

        min_mover = rosetta.protocols.minimization_packing.MinMover(mmap, scorefxn_min,
                                                                    'lbfgs_armijo_nonmonotone', 0.00001, True)
        min_mover.max_iter(max_iter)
        min_mover.cartesian(True)
        print("minimize...")
        min_mover.apply(pose)

    except:
        print('!!! idealization failed !!!')

    name = init_pdb[:-4] + f'_ref{max_iter}.pdb'
    pose.dump_pdb(name)
