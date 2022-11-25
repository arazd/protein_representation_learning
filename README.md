# Protein Representation Learning from Single-cell Microscopy Data

<!--![Build status](https://img.shields.io/github/workflow/status/bowang-lab/BIONIC/Python%20package)
![Version](https://img.shields.io/github/v/release/bowang-lab/BIONIC)
![Top language](https://img.shields.io/github/languages/top/bowang-lab/BIONIC)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6762584.svg)](https://doi.org/10.5281/zenodo.6762584)
-->
![License](https://img.shields.io/github/license/arazd/protein_representation_learning)

In this work, we explored different methods for protein representation learning from microscopy data. We evaluated the extracted representations on four biological benchmarks - subcellular compartments, biological processes, pathways and protein complexes.

**Check out our [paper (ICLR 2022, MLDD workshop)](https://arxiv.org/abs/2205.11676)!**


## About
Despite major developments in molecular representation learning, **extracting functional information from biological images** remains a non-trivial
computational task. In this work, we revisit deep learning models used for *classifying major subcellular localizations*, and evaluate
*representations extracted from their final layers*. We show that **simple convolutional networks trained on localization classification can learn protein representations that encapsulate diverse functional information**, and significantly outperform currently used autoencoder-based models. 

## Methods & Results

## How to run


## References 

If you found this work useful for your research, please cite:

Razdaibiedina A, Brechalov A. Learning multi-scale functional representations of proteins from single-cell microscopy data. InICLR2022 Machine Learning for Drug Discovery 2022 Mar 31.

```
@article{razdaibiedina2022learning,
  title={Learning multi-scale functional representations of proteins from single-cell microscopy data},
  author={Razdaibiedina, Anastasia and Brechalov, Alexander},
  journal={arXiv preprint arXiv:2205.11676},
  year={2022}
}
```

The supervised model we used for representation learning was first introduced in this paper:

Kraus OZ, Grys BT, Ba J, Chong Y, Frey BJ, Boone C, Andrews BJ. Automated analysis of high‐content microscopy data with deep learning. Molecular systems biology. 2017 Apr;13(4):924.

```
@article{kraus2017automated,
  title={Automated analysis of high-content microscopy data with deep learning},
  author={Kraus, Oren Z and Grys, Ben T and Ba, Jimmy and Chong, Yolanda and Frey, Brendan J and Boone, Charles and Andrews, Brenda J},
  journal={Molecular systems biology},
  volume={13},
  number={4},
  pages={924},
  year={2017}
}
```

