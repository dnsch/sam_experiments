# Experiments with Sharpness Aware Minimization (SAM)

This repository contains code for experiments exploring the relationship between [Sharpness Aware Minimization (SAM)](https://arxiv.org/abs/2010.01412) methods and generalization error. The code allows you to test multiple model architectures, regularization techniques, and datasets to evaluate the effectiveness of SAM.

---

## Instructions

#### Setting Up the Virtual Environment

To run the code, you must first set up a Python environment with the required dependencies specified in 📄 [`requirements.txt`](requirements.txt). The code was tested with **Python 3.12.11**.

A common way to set up an appropriate environment is using the python `venv` command:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
pip install -r requirements.txt
```

The 📄 [`requirements.txt`](requirements.txt) includes packages that help with downloading the datasets
and those that are needed to plot the loss surface using [loss-landscape](https://github.com/tomgoldstein/loss-landscape).

Other virtual environment tools include pyenv and uv, these also allow you to specify a Python version and might be faster.

---

#### Downloading the Data

To download all necessary datasets (used for training and testing SAMformer and CIFAR-10 for VGG models), run the following command inside the virtual environment:

```bash
python scripts/download_data.py
```

To download specific datasets:

* ##### Download SAMformer dataset

```bash
python scripts/download_samformer_dataset.py
```

* ##### Download CIFAR-10 dataset

```bash
python scripts/download_cifar10_dataset.py
```

If the scripts do not work, you can download the datasets manually using the following links:

[SAMformer datasets (Google Drive)](https://drive.google.com/uc?id=1alE33S1GmP5wACMXaLu50rDIoVzBM4ik)

[CIFAR-10 dataset (Official Link)](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

Once downloaded, place the SAMformer ```.csv``` files and the 📁 ```cifar-10-batches-py``` folder into the 📁 ```data/samformer_datasets``` and 📁 ```data/cifar10_datasets``` directories, respectively.

---

#### Running the Code

Once the data is downloaded and placed in the correct directories, you can run the experiments.

The general command to train and test a model is:

```bash
python code/experiments/[model]/main.py
```

where `[model]` refers to the model you want to use.

For example, to run the **SAMformer** model with the **ETTh1** dataset and seed `1`, use:

```bash
python code/experiments/samformer/main.py --dataset ETTh1 --seed 1
```

To run the same experiment *without SAM*, append the `--no_sam` flag:

```bash
python code/experiments/samformer/main.py --dataset ETTh1 --seed 1 --no_sam
```

This script will generate training and testing statistics plots. The output includes:

* A log file containing the training configuration and progress.
* A 📁 `saved_models/` folder containing the trained model for each epoch.
* A 📄 `final_model_s[seed].pt` file – the best model (i.e., the one with the lowest validation error).

All outputs are saved in the 📁 [`results/`](results/) directory. Files are named according to model type and arguments to help distinguish between experiment configurations.

---

##### Viewing Available Arguments

To view all available options for a given experiment, use the `--help` flag:

```bash
python code/experiments/[model]/main.py --help
```

---

#### Exploring the Loss Landscape

This repository includes a modified version of [loss-landscape](https://github.com/tomgoldstein/loss-landscape) for approximations of the loss surface.

To generate a loss surface plot in a `1x1` region around a trained model’s local minimum with a resolution of `20x20`, run:

```bash
python extra/loss_landscape/plot_surface.py --mpi --cuda --x=-1:1:20 --y=-1:1:20 \
--vmax=0.5 --vlevel=0.01 \
--model_file results/samformer/ETTh1/seed_2024_seq_len_512_pred_len_96_bs_256_rho_0.5/final_model_s2023.pt \
--dir_type weights --xnorm filter --xignore biasbn \
--ynorm filter --yignore biasbn --plot \
--dataset samformer_datasets --model samformer \
--loss_name mse --dataset_name ETTh1
```

The resulting plot will be saved in the 📁 [`plots/loss_surface/`](plots/loss_surface/) directory.

Use the `--vmax` flag to cap the maximum loss value, which is helpful when comparing different landscapes. For an explanation of all available options, use the `--help` flag.

All argument parsing is handled in 📄 [`extra/loss_landscape/plot_surface.py`](extra/loss_landscape/plot_surface.py), which is a modified version of the original. Outdated functions (mostly related to MPI) were updated to ensure compatibility. However, this version was **not tested** with MPI or multi-GPU setups, so some original functionality may not be preserved.

---

## Repository Structure

```
.
├── 📁 code/       # Experiment logic, training routines, model code  
├── 📁 data/       # Downloaded datasets  
├── 📁 extra/      # External repositories  
├── 📁 results/    # Trained models and experiment outputs  
└── 📁 scripts/    # Dataset download scripts  
```

---

### 📁 [`code/`](code/)

Inspired by [LargeST](https://github.com/liuxu77/LargeST), this folder includes:

* 📁 [`code/experiments/`](code/experiments/): contains subdirectories for each model, each with its own `main.py` to launch experiments.
* 📁 [`code/src/`](code/src/): contains core engine and model files:
  * 📁 [`code/src/base/`](code/src/base/): includes base classes:
    * 📄 [`engine.py`](code/src/base/engine.py)
    * 📄 [`model.py`](code/src/base/model.py)
  * 📁 [`code/src/engines/`](code/src/engines/): contains custom training/test logic for each model.
  * 📁 [`code/src/models/`](code/src/models/): defines the model architectures.

The files [`samformer_engine.py`](code/src/engines/samformer_engine.py) and [`samformer.py`](code/src/models/samformer.py) are modified versions from the original [SAMformer PyTorch implementation](https://github.com/romilbert/samformer/tree/main/samformer_pytorch).

To add a new model:

1. Create a custom 📄 `[model]_engine.py` in [`engines/`](code/src/engines/) that inherits from [`base/engine.py`](code/src/base/engine.py).
2. Add a 📄 `[model].py` in [`models/`](code/src/models/) that inherits from [`base/model.py`](code/src/base/model.py).
3. Create a 📁 `experiments/[model]/` folder with a 📄 `main.py`.

---

### 📁 [`data/`](data/)

This folder stores the downloaded datasets:

* [SAMformer Datasets (Google Drive)](https://drive.google.com/uc?id=1alE33S1GmP5wACMXaLu50rDIoVzBM4ik)
* [CIFAR-10 Dataset (Official Site)](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

---

### 📁 [`extra/`](extra/)

Contains additional repositories used in this project:

* [loss-landscape](https://github.com/tomgoldstein/loss-landscape)
* [SAM PyTorch implementation](https://github.com/davda54/sam)

---

### 📁 [`results/`](results/)

This directory stores:

* Trained model checkpoints
* Plots generated from training/testing runs

Each run is saved in a separate folder named after the model and its configuration.

---

### 📁 [`scripts/`](scripts/)

Contains helper scripts to download the SAMformer and CIFAR-10 datasets automatically.

---
