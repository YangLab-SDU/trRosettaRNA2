# trRosettaRNA2: Predicting RNA 3D structure and conformers using a pre-trained secondary structure model and structure-aware attention

[![Python version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)  [![PyTorch version](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square)](https://pytorch.org/) [![PyTorch_Geometric: 2020+](https://img.shields.io/badge/PyTorch_Geometric-2.0%2B-yellow?style=flat-square)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) [![PyRosetta: 2020+](https://img.shields.io/badge/PyRosetta-2020%2B-green?style=flat-square) ](https://www.pyrosetta.org/downloads)

![overview.png](https://github.com/quailwwk/trRosettaRNA2/blob/main/example/figures/overview.png?raw=true)

This package is a part of trRosettaRNA2, a deep learning-based RNA structure prediction protocol. 

Starting from an MSA, the trRosettaRNA2 pipeline comprises the following steps: 

 - Secondary structure (SS) prediction using a transformer network
 - 3D structure prediction using an end-to-end neural network.
 - (optional) 3D structure folding by energy minimization.

For more information about the trRosettaRNA2 pipeline, please refer to the first subsection of the METHODS section in the manuscript.

We also establish a user-friendly [webserver](http://yanglab.qd.sdu.edu.cn/trRosettaRNA/) for trRosettaRNA2.



Installation
----
### Step 1. Clone the repository

```bash
git clone https://github.com/quailwwk/trRosettaRNA2.git
cd trRosettaRNA2
```

### Step 2. Environment installation

It is recommended to use `mamba` to manage the Python dependencies, which can be installed following [Mamba Installation — documentation](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). 

Once the `mamba` is installed, a new environment can be created and activated: 

```bash
### tested on our devises with CUDA 12.4 and 11.2 ###
mamba env create -f environment.yml
```

### Step 3. Download the network weights

```bash
wget http://yanglab.qd.sdu.edu.cn/trRosettaRNA/download/params_trRNA2.tar.bz2
tar -jxvf params_trRNA2.tar.bz2
```



Usage
----

### Step 1. prepare inputs

The primary input for trRosettaRNA2 is a Multiple Sequence Alignment (MSA) of the target RNA. This MSA can be generated using homology search tools such as [Infernal](http://eddylab.org/infernal/) and [BLASTN](https://blast.ncbi.nlm.nih.gov/). Recommended sequence databases for this search include [NT](https://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/) and [RNAcentral](https://rnacentral.org/). Before prediction, the MSA file **must be converted** into A3M format (see details at https://yanglab.qd.sdu.edu.cn/trRosettaRNA/msa_format.html) 

Optionally, a single sequence can be used instead of the MSA. This approach is faster but may decrease prediction accuracy.

### Step 2. run prediction

**An basic example for prediction:**

```bash
python -m trRNA2.predict -i example/seq.a3m -o example/output
```

This command executes the default trRosettaRNA2 prediction procedure. This process utilizes its internal secondary structure (SS) module, `trRNA2-SS`, to predict SS and performs **end-to-end** structure prediction. The predicted 3D structure will be saved as a PDB file `model_1_ref200.pdb` under the `-o` directory.

**Alternative prediction configurations:**

You can optionally run predictions using alternative configurations. Examples include:

- **run the PyRosetta version**

  ```bash
  python -m trRNA2.predict -i example/seq.a3m -o example/output -pyrosetta -fas example/seq.fasta
  ```

  In this mode, predicted geometric restraints are converted into energy terms. These terms, combined with the Rosetta energy function, guide the 3D structure refinement process via energy minimization. The predicted 3D structure will be saved as a PDB file `model_1_pyrosetta.pdb` under the `-o` directory.

- **Using your custom SS:**

  ```bash
  python -m trRNA2.predict -i example/seq.a3m -o example/output -ss example/seq.dbn -ss_fmt dot_bracket 
  ```

  Supported SS format:  `bpseq`, `dot_bracket`, `ct`, or `prob` (txt file storing the base-pairing probability matrix)

For a complete description of all `trRNA2.predict` options and arguments, please run:

```bash
python -m trRNA2.predict -h
```



## Citation 

If you use trRosettaRNA2 in your research or work, please cite our publication: 

```
@article {Wang2024trRosettaRNA2,
	title = {Predicting RNA 3D structure and conformers using a pre-trained secondary structure model and structure-aware attention},
	author = {Wenkai Wang, Zhenling Peng, and Jianyi Yang},
	journal = {bioRxiv},
	year = {2025},
	doi = {10.1101/2025.04.09.647915}
}
```



## Questions & Issues

Have questions or encountered an issue with trRosettaRNA2? Please [open an issue](https://github.com/quailwwk/trRosettaRNA2/issues/new) in our GitHub repository. We'll do our best to help!

