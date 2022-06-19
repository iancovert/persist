# PERSIST

**Predictive and robust gene selection for spatial transcriptomics (PERSIST)** is a computational approach to select a small number of target genes for FISH experiments. PERSIST relies on a reference scRNA-seq dataset, and it uses deep learning to identify genes capable of reconstructing the genome-wide expression profile, or alternatively any other target of interest (e.g., transcriptomic cell types). PERSIST binarizes gene expression levels during the selection process to account for the measurement shift between scRNA-seq and FISH expression counts.

You can install the package into your Python environment by cloning the repository and then pip installing it as follows:

```bash
pip install .
```

The only software dependencies for this package are PyTorch, numpy, tqdm, sklearn and h5py, and our implementation is compatible with the most recent versions of all these packages.

See [here](https://github.com/iancovert/persist/blob/main/notebooks/demo.ipynb) for a demo notebook. The data is too large to upload on GitHub, but it can be downloaded from Google Drive [here](https://drive.google.com/file/d/1uEAXIyU58cXZEvUqM3FV-ezblH2c4Waf/view?usp=sharing).
