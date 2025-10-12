# Tutorial for trRosettaRNA2 training

## 前置准备

### Downloading training set files

```bash
bash scripts/download_training_set.sh
```

### Environment setup

If you have installed the environment for the older version of trRosettaRNA2 (before training code incorporate), two additional dependencies should be installed before performing training:

```bash
pip install tqdm openpyxl
```

## Training

The main file for training is the `train.py` script, which can be run as follows:

```bash
python -m trRNA2.train \
	-fas data/fasta_files/ \
	-npz data/npz_files/ \
	-out retrain/ \
	-gpu 0
```

This will automatically train the trRosettaRNA2 model following the three stages mentioned in the paper. The trained models, log file, and validation metrics will be appeared under `retrain/` directory.

It's recommended to run the training  script on a GPU with 80GB memory. If memory issue exists, please reduce the `-crop_size` argument.