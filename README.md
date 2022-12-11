# PERSIST

**Predictive and robust gene selection for spatial transcriptomics (PERSIST)** is a computational approach to select target genes for FISH studies. PERSIST relies on a reference scRNA-seq dataset, and it uses deep learning to identify genes that are predictive of the genome-wide expression profile or any other target of interest (e.g., transcriptomic cell types). PERSIST also binarizes gene expression levels during the selection process, which helps account for the measurement shift between expression counts obtained by scRNA-seq and FISH.

## Installation

You can install the package by cloning the repository and pip installing it as follows:

```bash
pip install -e .
```

Please be sure to install a version of [PyTorch](https://pytorch.org/get-started/locally/) that is compatible with your GPU (if applicable).

## Usage

PERSIST is designed to offer flexibility while requiring minimal tuning. For a demonstration of how it's used, from data preparation through gene selection, please see the following Jupyter notebooks:

- [00_data_proc.ipynb](https://github.com/iancovert/persist/blob/main/notebooks/00_data_proc.ipynb) shows how to download and pre-process one of the datasets used in our paper, the VISp SmartSeq v4 dataset from [Tasic et al., 2018](https://www.nature.com/articles/s41586-018-0654-5)
- [01_persist_supervised.ipynb](https://github.com/iancovert/persist/blob/main/notebooks/01_persist_supervised.ipynb) shows how to use PERSIST to select genes that are maximally predictive of cell type labels
- [02_persist_unsupervised.ipynb](https://github.com/iancovert/persist/blob/main/notebooks/02_persist_unsupervised.ipynb) shows how to use PERSIST to select genes that are maximally predictive of the genome-wide expression profile
